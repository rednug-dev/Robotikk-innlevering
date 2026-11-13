"""Markorfolging med LiDAR-hindringsstopp, versjon 2026-11-12.4-status.

Uten --drive: avlesning, motorene aktiveres ikke.
Standard: folgeavstand 0.60 m, maks 0.16 m/s, kjoretid 60 s.
Terminal: KJØRER/STOPP, markoravstand, kommandert sving og positive LiDAR-returer.
--data/--verbose beholdes for gamle kommandoer; terminalen har alltid samme fire felt.
--video lagrer AVI i logs/ med gronn markorramme og KJØR/STOPP.
Fullstendige stoppgrunner beholdes i CSV; andre feil i logs/marker_follower_errors.log.
Stopp ved hindring <=0.40 m foran; gjenstart krever klaring >=0.50 m.
For hver kommando kontrolleres arealet roboten kan na: 0.15 m frem og/eller
20 grader sving. Er hele buen fri, kjorer og svinger den samtidig; ellers
svinger den pa stedet. Ingen omkjoring.
Retninger uten lidarretur er ukjente og gir pause som standard.
--assume-visible-obstacles endrer det; flagget er en eksplisitt
deteksjonsantakelse, ikke et visningsflagg.
Markortap: stopp, ett sok ca. +/-20 grader (maks 15 s), deretter vent.
Sokevinkelen er et estimat fra sendte kommandoer; test fysisk.
Kalibrering: calibration_pi.npz ved siden av skriptet, 640x480, markor ID 0.
"""
import argparse
import csv
import logging
import math
import os
import queue
import signal
import threading
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import serial
from dynamixel_sdk import COMM_RX_CORRUPT, COMM_RX_TIMEOUT, COMM_SUCCESS, PacketHandler, PortHandler

VERSION = "2026-11-12.4-status"
MARKER_SIZE_M = 0.1315
IMAGE_SIZE = (640, 480)
TARGET_DISTANCE_M = 0.60
STOP_DISTANCE_M = 0.40        # Nearest front return, measured from the LiDAR.
RESTART_DISTANCE_M = 0.50
CAMERA_MAX_AGE_S = 0.30
LIDAR_MAX_AGE_S = 0.65
VIDEO_FPS = 10
LOOKAHEAD_M = 0.15            # Area checked ahead of every forward command ...
LOOKAHEAD_TURN_DEG = 20       # ... and of every turn command.
MIN_OBSTACLE_WIDTH_M = 0.05   # Projected width in the LiDAR plane; an assumption.
LIDAR_WARN_RETURNS = 150      # Diagnostic warning only.
LIDAR_FLOOR_RETURNS = 36      # Below 10 % coverage nothing moves; a heuristic.
LIDAR_MAX_ZERO_RUN_DEG = 60   # Longer runs without returns always stay unknown.
LIDAR_MAX_COVERED_GAP_DEG = 3
LIDAR_STARTUP_SCANS = 10
FRONT_INDICES = tuple(range(330, 360))+tuple(range(30))  # Raw indices, about +/-30 degrees.
LIDAR_DEVICE = "/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0"
OPENCR_DEVICE = "/dev/serial/by-id/usb-ROBOTIS_OpenCR_Virtual_ComPort_in_FS_Mode_FFFFFFFEFFFF-if00"


class Decision(NamedTuple):
    forward: float
    yaw: float
    code: str               
    message: str
    searching: bool = False


def hold(code, message):
    return Decision(0.0, 0.0, code, message)


def diagnostic_note(message):
    """Technical details stay in a file, away from the operator display."""
    try:
        path = Path(__file__).resolve().parent / "logs" / "marker_follower_errors.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{datetime.now().isoformat(timespec='seconds')} {message}\n")
    except OSError:
        # Diagnostics must never prevent motor shutdown.
        pass


# ---------------------------------------------------------------- motors

class OpenCR:
    """TurtleBot3 OpenCR as a Dynamixel 2.0 device (ID 200), driven like the ROBOTIS node."""
    ID, MODEL_NUMBER = 200, 20480
    DEBUG, HEARTBEAT, TORQUE, CMD_VEL = 14, 19, 149, 150  # Control table addresses.
    MAX_SPEED_M_S = 0.16
    MAX_YAW_RAD_S = 0.30

    def __init__(self, device):
        self.port = PortHandler(device)
        self.packet = PacketHandler(2.0)
        self.device = device
        self.opened = self.enabled = False
        self.heartbeat = self.retries = 0

    def _write(self, address, data):
        """Acknowledged write; never TxOnly. Every write here sets a register to a
        value, so the identical packet is repeated once if only the reply was lost."""
        if not self.opened:
            raise RuntimeError("OpenCR-forbindelsen er ikke apen.")
        for attempt in range(2):
            result, error = self.packet.writeTxRx(self.port, self.ID, address, len(data), list(data))
            if result == COMM_SUCCESS and not error:
                self.retries += attempt
                return
            if attempt or error or result not in (COMM_RX_TIMEOUT, COMM_RX_CORRUPT):
                break
            time.sleep(0.005)
            self.port.clearPort()
        raise RuntimeError(f"Skrivefeil adresse {address}: "
                           f"{self.packet.getTxRxResult(result)}, enhetsfeil {error}")

    def _write_velocity(self, forward_m_s, yaw_rad_s):
        # One 24-byte command (linear xyz, angular xyz; 0.01 units), as in ROBOTIS cmd_vel_callback.
        values = (round(forward_m_s*100), 0, 0, 0, 0, round(yaw_rad_s*100))
        self._write(self.CMD_VEL, b"".join(value.to_bytes(4, "little", signed=True) for value in values))

    def connect(self):
        if not self.port.openPort():
            raise RuntimeError(f"Kunne ikke apne {self.device}.")
        self.opened = True
        try:
            if not self.port.setBaudRate(1000000):
                raise RuntimeError("Kunne ikke sette baudrate 1000000.")
            time.sleep(2)
            model, result, error = self.packet.ping(self.port, self.ID)
            if result != COMM_SUCCESS or error or model != self.MODEL_NUMBER:
                raise RuntimeError(f"Uventet OpenCR-svar: modell={model}, status={result}, feil={error}.")
            self.stop()
            self._write(self.DEBUG, [0])  # Debug off keeps the firmware's heartbeat watchdog active.
        except BaseException:
            self.close()
            raise

    def enable(self):
        self.stop()
        self._write(self.TORQUE, [1])
        self.enabled = True

    def set_velocity(self, forward_m_s, yaw_rad_s):
        if not self.enabled:
            raise RuntimeError("Motorene er ikke aktivert.")
        try:
            if not (math.isfinite(forward_m_s) and math.isfinite(yaw_rad_s)):
                raise ValueError("Hastigheter ma vaere endelige tall.")
            self._write_velocity(max(-self.MAX_SPEED_M_S, min(self.MAX_SPEED_M_S, forward_m_s)),
                                 max(-self.MAX_YAW_RAD_S, min(self.MAX_YAW_RAD_S, yaw_rad_s)))
            # The firmware zeroes the wheels when this counter stops changing for 0.5 s.
            self.heartbeat = (self.heartbeat+1) % 256
            self._write(self.HEARTBEAT, [self.heartbeat])
        except BaseException:
            try:
                self.stop()
            except Exception as stop_error:
                diagnostic_note(f"Stopp feilet. Bruk hovedbryteren: {stop_error}")
            raise

    def stop(self):
        """Attempt both zero velocity and torque-off, even if one of them fails."""
        self.enabled = False
        if not self.opened:
            return
        failures = []
        for action in (lambda: self._write_velocity(0.0, 0.0), lambda: self._write(self.TORQUE, [0])):
            try:
                action()
            except Exception as error:
                failures.append(str(error))
        if failures:
            raise RuntimeError("; ".join(failures))

    def close(self):
        """Stop and release the port. Never raises, also when the port was never opened."""
        if not self.opened:
            return
        for action, problem in ((self.stop, "Stopp feilet. Bruk hovedbryteren"),
                                (self.port.closePort, "Kunne ikke lukke OpenCR-porten")):
            try:
                action()
            except Exception as error:
                diagnostic_note(f"{problem}: {error}")
        self.opened = False


# ---------------------------------------------------------------- LiDAR data

def decode_packet(packet):
    """One 42-byte LDS-01 packet -> (block 0-59, six raw millimetres, six intensities, rpm)."""
    if len(packet) != 42 or packet[0] != 0xFA or not 0xA0 <= packet[1] <= 0xDB:
        raise ValueError("Ugyldig LDS-01-pakkehode.")
    # The manufacturer allows either byte 40 or 41 to match.
    if (0xFF-sum(packet[:40])) & 0xFF not in packet[40:42]:
        raise ValueError("LDS-01 checksum stemmer ikke.")
    # Per ray: intensity in bytes 0-1, distance in mm in bytes 2-3.
    rays = range(4, 40, 6)
    return (packet[1]-0xA0, [packet[i+2] | packet[i+3] << 8 for i in rays],
            [packet[i] | packet[i+1] << 8 for i in rays], (packet[2] | packet[3] << 8)/10)


@dataclass(frozen=True)
class Scan:
    raw_mm: tuple           
    intensities: tuple
    started_at: float       
    completed_at: float
    rpm: float
    # Added by LidarPolicy.prepare():
    zero_rules: tuple = (None,)*360
    quality_reason: str = ""
    quality_warning: str = ""
    stats: dict = field(default_factory=dict)


def range_reading(scan, index):
    """(metres or None, kind). Zero means no return and is never a distance."""
    mm = scan.raw_mm[index]
    if mm == 0:
        return None, "missing"
    if mm >= 65535:
        return None, "invalid"
    if mm < 120:
        return mm/1000, "too_close"  
    if mm <= 3500:
        return mm/1000, "valid"
    if scan.intensities[index] > 0:
        return 3.5, "long"
    return None, "long_unverified"


def circular_runs(flags):
    """Groups of consecutive true indices, joining a group that crosses 359/0."""
    if not any(flags):
        return ()
    if all(flags):
        return (tuple(range(len(flags))),)
    start = flags.index(False)
    groups, group = [], []
    for offset in range(1, len(flags)+1):
        index = (start+offset) % len(flags)
        if flags[index]:
            group.append(index)
        elif group:
            groups.append(tuple(group))
            group = []
    return tuple(groups)


class LidarPolicy:
    """How rays without a return may be used. Never fabricates millimetres or intensity."""

    def __init__(self, assume_visible=False, min_width_m=MIN_OBSTACLE_WIDTH_M):
        self.assume_visible, self.min_width_m = assume_visible, min_width_m
        self._raw = self._prepared = None

    def prepare(self, scan):
        """The scan plus zero rules, quality texts and log statistics (cached per scan)."""
        if scan is None:
            return None
        if scan is not self._raw:
            self._raw, self._prepared = scan, self._derive(scan)
        return self._prepared

    def _derive(self, scan):
        positive = [0 < mm < 65535 for mm in scan.raw_mm]
        count = sum(positive)
        longest = max((len(run) for run in circular_runs([not p for p in positive])), default=0)
        reason = warning = ""
        if count < LIDAR_FLOOR_RETURNS:
            reason = (f"LiDAR-kvalitet: svaert fa positive returer ({count}/360; "
                      f"reservegrense {LIDAR_FLOOR_RETURNS}); utilstrekkelig dekning")
        else:
            notes = []
            if count < LIDAR_WARN_RETURNS:
                notes.append(f"{count}/360 positive returer (varselsgrense {LIDAR_WARN_RETURNS})")
            if longest > LIDAR_MAX_ZERO_RUN_DEG:
                notes.append(f"hull {longest} grader; hull over {LIDAR_MAX_ZERO_RUN_DEG} forblir ukjente")
            warning = "; ".join(notes)
        rules = [None]*360
        if self.assume_visible and not reason:
            for run in circular_runs([mm == 0 for mm in scan.raw_mm]):
                if len(run) > LIDAR_MAX_ZERO_RUN_DEG:
                    continue  
                left = range_reading(scan, (run[0]-1) % 360)
                right = range_reading(scan, (run[-1]+1) % 360)
                if not {left[1], right[1]} <= {"valid", "long"}:
                    continue  
                if len(run) <= LIDAR_MAX_COVERED_GAP_DEG:
                    rule = ("covered", self.min_width_m/(2*math.tan(math.radians(len(run)+1)/2)))
                else:
                    rule = ("assumed", 3.5)
                rule += (min(left[0], right[0]),)
                for index in run:
                    rules[index] = rule
        tags = dict(measured=0, covered=0, assumed=0, unknown=0)
        for index in range(360):
            measured = range_reading(scan, index)[0] is not None
            tags["measured" if measured else rules[index][0] if rules[index] else "unknown"] += 1
        stats = dict(positive_returns=count, near_returns=sum(0 < mm <= 1500 for mm in scan.raw_mm),
                     longest_zero_run_deg=longest, **{"scan_"+key: value for key, value in tags.items()})
        return replace(scan, zero_rules=tuple(rules), quality_reason=reason,
                       quality_warning=warning, stats=stats)


def ray_permission(scan, index, required_range):
    """'covered'/'assumed' if policy lets a ray without a return count as free out to
    required_range. A bounded local permission, never a fake measurement."""
    rule = scan.zero_rules[index]
    if rule and required_range < rule[1] and required_range < rule[2]:
        return rule[0]
    return None


def front_summary(scan):
    """(nearest front range or None, rays with a usable return). Near-field counts as 0.0 m."""
    readings = [range_reading(scan, index) for index in FRONT_INDICES]
    good = [distance for distance, kind in readings if kind in ("valid", "long")]
    if any(kind == "too_close" for _, kind in readings):
        return 0.0, len(good)
    return (min(good) if good else None), len(good)


def front_block(scan, minimum):
    """Restart hysteresis with the same zero-return policy as travel: reason code or None."""
    unknown = False
    for index in FRONT_INDICES:
        distance, _kind = range_reading(scan, index)
        if distance is None:
            unknown = unknown or ray_permission(scan, index, minimum) is None
        elif distance < minimum:
            return "measured_obstacle"
    return "unknown_space" if unknown else None


def lidar_fault(scan, now):
    """(reason code, text) when the scan must not be used, else None."""
    if scan is None:
        return "stale_or_missing_data", "venter pa LiDAR: ingen komplett skanning tilgjengelig"
    if len(scan.raw_mm) != 360 or len(scan.intensities) != 360:
        return "invalid_sensor_data", f"LiDAR: ufullstendig skanning ({len(scan.raw_mm)}/360 straler)"
    if not scan.started_at <= scan.completed_at <= now:  # Also false for NaN.
        return "invalid_sensor_data", "LiDAR: ugyldig tidsrekkefolge (start <= slutt <= naa kreves)"
    age = now-scan.started_at
    if age > LIDAR_MAX_AGE_S:
        return "stale_or_missing_data", f"LiDAR-skanning er {age:.3f} s gammel; grense {LIDAR_MAX_AGE_S:.2f} s"
    if not 250 <= scan.rpm <= 350:
        return "invalid_sensor_data", f"LiDAR: RPM {scan.rpm:.1f} utenfor 250-350"
    return None


def camera_fault(frame, now):
    if frame is None:
        return "stale_or_missing_data", "venter pa kamera: ingen bilder tilgjengelig"
    if not frame.started_at <= now:
        return "invalid_sensor_data", "kamera: ugyldig tidsstempel"
    age = now-frame.started_at
    if age > CAMERA_MAX_AGE_S:
        return "stale_or_missing_data", f"kamerabilde er {age:.3f} s gammelt; grense {CAMERA_MAX_AGE_S:.2f} s"
    return None


class LDS01:
    ROTATION_BYTES = 60*42 

    def __init__(self, device=LIDAR_DEVICE):
        self.port = serial.Serial(device, baudrate=230400, timeout=0.05, write_timeout=0.2, exclusive=True)
        self.buffer = bytearray()
        self.expected = 0
        try:
            self.port.reset_input_buffer()
            self.port.write(b"b")
        except BaseException:
            self.port.close()
            raise

    def read_scan(self, timeout_s=3.0):
        deadline = time.monotonic()+timeout_s
        while time.monotonic() < deadline:
            if self.port.in_waiting > self.ROTATION_BYTES:
                self.port.reset_input_buffer()
                self.buffer.clear()
                self.expected = 0
            self.buffer.extend(self.port.read(min(self.ROTATION_BYTES, max(1, self.port.in_waiting))))
            while len(self.buffer) >= 42:
                try:
                    block, raw_mm, intensities, rpm = decode_packet(bytes(self.buffer[:42]))
                except ValueError:
                    del self.buffer[0]  
                    self.expected = 0
                    continue
                del self.buffer[:42]
                now = time.monotonic()
                if block == 0:
                    self.expected = 0
                    self.raw_mm, self.intensities, self.rpm_total, self.first_parsed = [], [], 0.0, now
                if block != self.expected:
                    self.expected = 0
                    continue
                self.raw_mm.extend(raw_mm)
                self.intensities.extend(intensities)
                self.rpm_total += rpm
                self.expected += 1
                if self.expected == 60:
                    self.expected = 0
                    rpm = self.rpm_total/60
                    period = 60/rpm if rpm > 0 else 0.2
                    queued = len(self.buffer)+self.port.in_waiting
                    completed = now-queued/self.ROTATION_BYTES*period
                    return Scan(tuple(self.raw_mm), tuple(self.intensities),
                                min(self.first_parsed, completed-period), completed, rpm)
        self.expected = 0
        self.buffer.clear()
        self.port.reset_input_buffer()
        raise TimeoutError("Ingen komplett LDS-01-rotasjon med gyldig checksum mottatt.")

    def close(self):
        try:
            self.port.write(b"e")
        finally:
            self.port.close()


@dataclass(frozen=True)
class RobotGeometry:
    length: float = 0.281
    width: float = 0.306
    chassis_x: float = -0.064
    lidar_x: float = -0.064
    margin: float = 0.05

    def corners(self, x=0.0, y=0.0, yaw=0.0):
        """Counter-clockwise footprint corners with the axle at (x, y)."""
        c, s = math.cos(yaw), math.sin(yaw)
        rear, front, half = self.chassis_x-self.length/2, self.chassis_x+self.length/2, self.width/2
        return [(x+c*px-s*py, y+s*px+c*py)
                for px, py in ((rear, -half), (front, -half), (front, half), (rear, half))]


def convex_hull(points):
    """Counter-clockwise convex hull (monotone chain)."""
    points = sorted(set(points))

    def half(sequence):
        chain = []
        for p in sequence:
            while len(chain) >= 2 and ((chain[-1][0]-chain[-2][0])*(p[1]-chain[-2][1])
                                       - (chain[-1][1]-chain[-2][1])*(p[0]-chain[-2][0])) <= 0:
                chain.pop()
            chain.append(p)
        return chain[:-1]
    return half(points)+half(reversed(points))


def ray_exits(origin, rays, polygon, margin=0.0):
    """Range at which each ray from origin leaves a convex CCW polygon grown by margin; 0 on a miss."""
    points = np.asarray(polygon, dtype=float)
    edges = np.roll(points, -1, axis=0)-points
    inside = (edges[:, 0]*(origin[1]-points[:, 1])-edges[:, 1]*(origin[0]-points[:, 0])
              + margin*np.hypot(edges[:, 0], edges[:, 1]))
    slope = rays[:, 1:2]*edges[:, 0]-rays[:, 0:1]*edges[:, 1]
    parallel = np.abs(slope) < 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        crossing = -inside/slope
    enter = np.max(np.where(~parallel & (slope > 0), crossing, 0.0), axis=1)
    leave = np.min(np.where(~parallel & (slope < 0), crossing, np.inf), axis=1)
    outside = np.any(parallel & (inside < -1e-10), axis=1)
    hit = ~outside & (leave >= enter-1e-10) & np.isfinite(leave)
    return np.where(hit, np.maximum(leave, 0.0), 0.0)


@lru_cache(maxsize=None)
def swept_requirements(geometry, forward, turn_sign):
    """Per raw LiDAR index: (needed, exit range) for one look-ahead.

    The look-ahead is every pose the robot can reach by driving at most LOOKAHEAD_M
    (if forward) while its heading stays between 0 and turn_sign*LOOKAHEAD_TURN_DEG.
    That contains straight driving, turning on the spot and every arc in between.
    needed: the swept area reaches past the robot's present outline along this ray.
    exit range: where the ray leaves the convex hull of the swept area plus margin.
    """
    distance = LOOKAHEAD_M if forward else 0.0
    headings = [math.radians(turn_sign*degrees) for degrees in range(0, LOOKAHEAD_TURN_DEG+1, 2)]
    if not turn_sign:
        headings = headings[:1]
    fan = [(0.0, 0.0)]+[(distance*math.cos(h), distance*math.sin(h)) for h in headings if forward]
    areas = [convex_hull([p for x, y in fan for p in geometry.corners(x, y, h)]) for h in headings]
    origin = (geometry.lidar_x, 0.0)
    angles = np.radians((359-np.arange(360)) % 360)
    rays = np.column_stack((np.cos(angles), np.sin(angles)))
    present = ray_exits(origin, rays, geometry.corners())
    reach = np.max([ray_exits(origin, rays, area) for area in areas], axis=0)
    safe = ray_exits(origin, rays, convex_hull([p for area in areas for p in area]), geometry.margin+0.001)
    return tuple((bool(after > before+1e-6), float(limit)) for before, after, limit in zip(present, reach, safe))


def motion_clearance(scan, geometry, forward, turn_sign):
    """Is the look-ahead free? -> (clear, reason code, text, how the needed rays were resolved)."""
    counts = dict(measured=0, covered=0, assumed=0, unknown=0)
    blocked, near_gap, unknown = 0, False, []
    for index, (needed, exit_range) in enumerate(swept_requirements(geometry, forward, turn_sign)):
        distance, _kind = range_reading(scan, index)
        if distance is not None:
            blocked += distance <= exit_range
            counts["measured"] += needed
        elif needed:
            permission = ray_permission(scan, index, exit_range)
            counts[permission or "unknown"] += 1
            if permission is None:
                unknown.append(index)
                rule = scan.zero_rules[index]
                if rule and rule[2] <= exit_range:
                    blocked, near_gap = blocked+1, True
    total = sum(counts.values())
    if blocked:
        return (False, "measured_obstacle", f"hindring i bevegelsesarealet ({blocked} straler)"
                + ("; naer malt naboretur ved hull" if near_gap else ""), counts)
    if unknown:
        return (False, "unknown_space", f"ukjent planlagt areal ({len(unknown)}/{total} nodvendige straler; "
                f"raavinkler {','.join(map(str, unknown[:8]))})", counts)
    return (True, "motion", f"plasskontroll ({total} nodvendige straler; malt {counts['measured']}, "
            f"nabo {counts['covered']}, antatt {counts['assumed']})", counts)


# ---------------------------------------------------------------- camera

@dataclass(frozen=True)
class Target:
    x: float            
    z: float            
    error_px: float


@dataclass(frozen=True)
class Frame:
    started_at: float   
    target: object


def estimate_pose(corners, matrix, distortion):
    half = MARKER_SIZE_M/2
    world = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], np.float64)
    measured = np.ascontiguousarray(corners, dtype=np.float64).reshape(4, 2)
    if not np.isfinite(measured).all():
        return None
    result = cv2.solvePnPGeneric(world, measured, matrix, distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    candidates = list(zip(result[1], result[2])) if result[0] else []
    ok, rotation, translation = cv2.solvePnP(world, measured, matrix, distortion, flags=cv2.SOLVEPNP_ITERATIVE)
    if ok:
        candidates.append((rotation, translation))
    best = None
    for rotation, translation in candidates:
        if not (np.isfinite(rotation).all() and np.isfinite(translation).all()):
            continue
        rotation_matrix, _ = cv2.Rodrigues(rotation)
        if np.any((world @ rotation_matrix.T+translation.reshape(1, 3))[:, 2] <= 0):
            continue  
        predicted, _ = cv2.projectPoints(world, rotation, translation, matrix, distortion)
        error = float(np.sqrt(np.mean(np.sum((predicted.reshape(4, 2)-measured)**2, axis=1))))
        if best is None or error < best[1]:
            best = (translation.reshape(3), error)
    return best


def load_calibration():
    with np.load(Path(__file__).resolve().parent/"calibration_pi.npz", allow_pickle=False) as data:
        matrix = data["camera_matrix"].astype(np.float64)
        distortion = data["dist_coeffs"].astype(np.float64)
        if tuple(data["image_size"]) != IMAGE_SIZE:
            raise RuntimeError("Kalibreringen ma vaere for 640 x 480 bilder.")
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all() or not np.isfinite(distortion).all()
            or matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or distortion.size not in (4, 5, 8, 12, 14)):
        raise RuntimeError("Ugyldige kalibreringstall.")
    return matrix, distortion


# ---------------------------------------------------------------- controller

class Follower:
    """Follow marker ID 0 at a set distance, stop for obstacles, search once when it is lost."""
    SEARCH_PAUSE_S = 0.4
    SEARCH_YAW_RAD_S = 0.25
    SEARCH_LEG_RAD = math.radians(20)
    SEARCH_MAX_WALL_S = 15.0

    def __init__(self, geometry, max_speed=0.16, use_lidar=True, camera_left_m=0.064,
                 target_distance_m=TARGET_DISTANCE_M):
        self.geometry, self.max_speed, self.use_lidar = geometry, max_speed, use_lidar
        self.camera_left_m, self.target_distance_m = camera_left_m, target_distance_m
        self.report = {}            
        self.blocked = False        
        self.had_target = False
        self.last_bearing = 0.0
        self.lost_at = None         
        self.search_done = False
        self.search_angle = 0.0     
        self._stamp = None
        self._search_yaw = 0.0

    def record_command(self, now, yaw, searching):
        """Main calls this once a command has been acknowledged; yaw is 0 if no motor ran it."""
        self._integrate_search(now)
        self._stamp, self._search_yaw = now, (round(yaw*100)/100 if searching else 0.0)

    def _integrate_search(self, now):
        if self._stamp is not None:
            self.search_angle += abs(self._search_yaw)*max(0.0, now-self._stamp)
            self._stamp = now

    def choose(self, frame, scan, now):
        self._integrate_search(now)
        self.report = {}
        decision = self._decide(frame, scan, now)
        if not self.use_lidar or not (decision.forward or decision.yaw):
            return decision
        forward, turn = decision.forward > 0, (decision.yaw > 0)-(decision.yaw < 0)
        clear, code, text, self.report = motion_clearance(scan, self.geometry, forward, turn)
        if clear:
            return decision
        if forward and turn:
            arc_text = text
            clear, code, text, self.report = motion_clearance(scan, self.geometry, False, turn)
            if clear:
                return decision._replace(forward=0.0, message=f"{decision.message} | svinger pa stedet: {arc_text}")
        return hold(code, "PAUSE: "+text)

    def _decide(self, frame, scan, now):
        lidar_text = "LiDAR AV: ingen hindringsstopp"
        if self.use_lidar:
            fault = lidar_fault(scan, now)
            if fault:
                return hold(fault[0], "STOPP: "+fault[1])
            if scan.quality_reason:
                return hold("lidar_quality", "PAUSE: "+scan.quality_reason)
            nearest, valid = front_summary(scan)
            if nearest is not None and nearest <= STOP_DISTANCE_M:
                self.blocked = self.search_done = True
                return hold("measured_obstacle", f"STOPP: hindring {nearest:.2f} m foran")
            if self.blocked:
                code = front_block(scan, RESTART_DISTANCE_M)
                if code:
                    return hold(code, f"STOPP: venter pa malt klaring >={RESTART_DISTANCE_M:.2f} m"
                                + (" (retninger uten retur)" if code == "unknown_space" else ""))
                self.blocked = False
            lidar_text = (f"foran {'ukjent' if nearest is None else format(nearest, '.2f')+' m'} | "
                          f"LiDAR {valid}/{len(FRONT_INDICES)}")
        fault = camera_fault(frame, now)
        if fault:
            return hold(fault[0], "STOPP: "+fault[1])
        target = frame.target
        if target is None:
            return self._search(now)
        if not (all(math.isfinite(value) for value in (target.x, target.z, target.error_px))
                and 0.12 <= target.z <= 3.5 and 0 <= target.error_px <= 2.0):
            return hold("marker_or_state_wait", "STOPP: ugyldig markorposisjon")
        bearing = math.atan2(target.x-self.camera_left_m, target.z)
        self.had_target, self.last_bearing = True, bearing
        self.lost_at, self.search_done, self.search_angle, self._search_yaw = None, False, 0.0, 0.0
        yaw = 0.0 if abs(bearing) <= math.radians(3) else max(-0.30, min(0.30, -1.2*bearing))
        forward = 0.0
        if target.z > self.target_distance_m+0.05 and abs(bearing) <= math.radians(15):
            forward = min(self.max_speed, 0.45*(target.z-self.target_distance_m))
        return Decision(forward, yaw, "marker_or_state_wait",
                        f"FOLGER: markor {target.z:.2f} m | retning {math.degrees(bearing):+.1f} grader | {lidar_text}")

    def _search(self, now):
        wait = "marker_or_state_wait"
        if not self.had_target:
            return hold(wait, "STOPP: vis markor ID 0 forst")
        if self.search_done:
            return hold(wait, "STOPP: sok avsluttet; venter pa markor ID 0")
        if self.lost_at is None:
            self.lost_at = now
        elapsed = now-self.lost_at
        if elapsed >= self.SEARCH_MAX_WALL_S:
            self.search_done = True
            return hold(wait, "STOPP: sokets ventetid utlop; venter pa markor ID 0")
        if elapsed < self.SEARCH_PAUSE_S:
            return hold(wait, "STOPP: markor tapt; stanser for sok")
        legs = self.search_angle/self.SEARCH_LEG_RAD
        if legs >= 4:
            self.search_done = True
            return hold(wait, "STOPP: sok avsluttet; venter pa markor ID 0")
        toward = -1.0 if self.last_bearing >= 0 else 1.0
        yaw = (-toward if 1 <= legs < 3 else toward)*self.SEARCH_YAW_RAD_S
        return Decision(0.0, yaw, wait, f"SOKER {'VENSTRE' if yaw > 0 else 'HOYRE'}: ingen fremdrift", True)


def ramp(target, previous, step):
    """Rate-limit increases; reductions are immediate and a reversal passes through zero."""
    if target == 0 or target*previous < 0:
        return 0.0
    if abs(target) <= abs(previous):
        return target
    return math.copysign(min(abs(target), abs(previous)+step), target)


# ---------------------------------------------------------------- sensor threads

class Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = self.scan = self.failure = None

    def fail(self, source, error):
        with self.lock:
            if self.failure is None:
                self.failure = f"{source}: {error}"

    def snapshot(self):
        with self.lock:
            return self.frame, self.scan, self.failure


def draw_video_frame(image, outline, target, moving, elapsed):
    """Draw on a copy; the detector and raw camera image remain untouched."""
    image = image.copy()
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    if target is not None:
        points = np.asarray(outline, np.float64).reshape(-1, 1, 2).round().astype(np.int32)
        cv2.polylines(image, [points], True, (0, 255, 0), 3)
        cv2.putText(image, "MARKOR ID 0", (10, 28), font, scale, (0, 255, 0), thickness)
    else:
        cv2.putText(image, "INGEN MARKOR", (10, 28), font, scale, (0, 0, 255), thickness)
    distance = f"{target.z:.2f} m" if target is not None else "--"
    text = f"{'KJOR' if moving else 'STOPP'} | Distanse {distance}"
    cv2.putText(image, text, (10, 58), font, scale, (255, 255, 255), thickness)
    if moving:
        # Hershey fonts lack Ø. Add the slash to the O to render KJØR.
        prefix_width = cv2.getTextSize("KJ", font, scale, thickness)[0][0]-thickness
        (letter_width, letter_height), _ = cv2.getTextSize("O", font, scale, thickness)
        x = 10+prefix_width
        cv2.line(image, (x+3, 58), (x+letter_width-5, 58-letter_height+2),
                 (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(image, f"{elapsed:6.1f} s", (10, 468), font, 0.5, (255, 255, 255), 1)
    return image


class VideoLog:
    def __init__(self, path):
        self.path = path
        self.frames = queue.Queue(maxsize=3)
        self.moving = False

    def add(self, started, image, outline, target):
        try:
            self.frames.put_nowait((started, image, outline, target, self.moving))
        except queue.Full:
            pass

    def run(self, shutdown):
        writer = first = None
        written = 0
        try:
            while not (shutdown.is_set() and self.frames.empty()):
                try:
                    started, image, outline, target, moving = self.frames.get(timeout=0.2)
                except queue.Empty:
                    continue
                if writer is None:
                    writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"MJPG"), VIDEO_FPS, IMAGE_SIZE)
                    if not writer.isOpened():
                        raise RuntimeError(f"kunne ikke apne {self.path}")
                    first = started
                image = draw_video_frame(image, outline, target, moving, started-first)
                due = int((started-first)*VIDEO_FPS)+1
                for _ in range(min(due-written, 2*VIDEO_FPS)):
                    writer.write(image)
                written = max(written, due)
        except Exception as error:
            diagnostic_note(f"VIDEO: opptaket stoppet: {error}")
        finally:
            if writer is not None:
                writer.release()


def camera_worker(shared, shutdown, matrix, distortion, video=None):
    camera = None
    try:
        from picamera2 import Picamera2
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), parameters)
        camera = Picamera2()
        camera.configure(camera.create_preview_configuration(
            main={"size": IMAGE_SIZE, "format": "RGB888"}, queue=False))
        camera.start()
        if shutdown.wait(2.0):
            return
        while not shutdown.is_set():
            started = time.monotonic()
            image = camera.capture_array()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            found = [] if ids is None else ids.flatten().tolist()
            target = None
            if found.count(0) == 1:
                pose = estimate_pose(corners[found.index(0)], matrix, distortion)
                if pose is not None:
                    (x, _y, z), error = pose
                    target = Target(float(x), float(z), float(error))
            with shared.lock:
                shared.frame = Frame(started, target)
            if video is not None:
                video.add(started, image, corners[found.index(0)] if target else None, target)
            shutdown.wait(0.01)
    except Exception as error:
        shared.fail("Kamera", error)
    finally:
        if camera is not None:
            try:
                camera.close()
            except Exception as error:
                shared.fail("Lukking av kamera", error)


def lidar_worker(shared, shutdown):
    lidar = None
    try:
        lidar = LDS01()
        while not shutdown.is_set():
            try:
                scan = lidar.read_scan(timeout_s=0.6)
            except TimeoutError:
                scan = None  
            with shared.lock:
                shared.scan = scan
    except Exception as error:
        shared.fail("LiDAR", error)
    finally:
        if lidar is not None:
            try:
                lidar.close()
            except Exception as error:
                shared.fail("Lukking av LiDAR", error)


# ---------------------------------------------------------------- logging and main

class ControlLog:
    """Control decisions as CSV; not a replacement for lidar_log.py's raw packets."""
    COLUMNS = ("elapsed_s", "phase", "drive_enabled", "state", "reason_code", "reason",
               "forward_m_s", "yaw_rad_s", "scan_started_at", "positive_returns", "near_returns",
               "longest_zero_run_deg", "scan_measured", "scan_covered", "scan_assumed", "scan_unknown",
               "area_measured", "area_covered", "area_assumed", "area_unknown", "width_m", "assume_visible",
               "quality_warning", "sensor_checked_at", "scan_completed_at", "scan_duration_s",
               "lidar_age_s", "lidar_rpm", "camera_started_at", "camera_age_s")

    def __init__(self, path, policy):
        self.path = Path(path) if path else (Path(__file__).resolve().parent/"logs"
                                             / f"marker_follower_{datetime.now():%Y%m%d_%H%M%S_%f}.csv")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("x", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.COLUMNS)
        self.writer.writeheader()
        self.fixed = dict(width_m=policy.min_width_m, assume_visible=int(policy.assume_visible))
        self.last_flush = -math.inf

    def write(self, elapsed, phase, enabled, state, code, reason, forward, yaw, scan, report,
              now=None, frame=None):
        """now is when the sensor ages were checked for this decision."""
        row = dict(self.fixed, elapsed_s=f"{elapsed:.6f}", phase=phase, drive_enabled=int(enabled),
                   state=state, reason_code=code, reason=reason, forward_m_s=forward, yaw_rad_s=yaw)
        row.update({"area_"+key: value for key, value in report.items()})
        if scan is not None:
            row.update(scan.stats, scan_started_at=scan.started_at, scan_completed_at=scan.completed_at,
                       scan_duration_s=scan.completed_at-scan.started_at, lidar_rpm=scan.rpm,
                       quality_warning=scan.quality_warning)
        if frame is not None:
            row["camera_started_at"] = frame.started_at
        if now is not None:
            row["sensor_checked_at"] = now
            if scan is not None:
                row["lidar_age_s"] = now-scan.started_at
            if frame is not None:
                row["camera_age_s"] = now-frame.started_at
        self.writer.writerow(row)
        if elapsed-self.last_flush >= 1.0:
            self.file.flush()
            self.last_flush = elapsed

    def close(self):
        self.file.close()


def startup_lidar(shared, shutdown, policy, logger, started, verbose):
    """Log ten distinct fresh scans with torque off (max 12 s) -> (good-quality streak, newest stamp)."""
    count = streak = 0
    stamp, problem = -math.inf, "venter pa LiDAR"
    last_print = -math.inf
    while not shutdown.is_set() and count < LIDAR_STARTUP_SCANS:
        frame, scan, failure = shared.snapshot()
        now = time.monotonic()
        if failure:
            raise RuntimeError(failure)
        if now >= started+12.0:
            raise TimeoutError(f"Oppstart: fikk {count}/{LIDAR_STARTUP_SCANS} ferske LiDAR-skanninger; {problem}.")
        fault = lidar_fault(scan, now)
        if fault or scan.started_at <= stamp:
            problem = fault[1] if fault else "ingen ny komplett skanning"
            shutdown.wait(0.03)
            continue
        scan = policy.prepare(scan)
        stamp, count = scan.started_at, count+1
        streak = 0 if scan.quality_reason else streak+1
        message = scan.quality_reason or "oppstart: avlesning mottatt; bevegelsesarealet kontrolleres separat"
        if scan.quality_warning:
            message += "; dekningsvarsel: "+scan.quality_warning
        if now-last_print >= 0.5:
            print(status_line(0.0, 0.0, frame, scan, now), flush=True)
            last_print = now
        logger.write(now-started, "startup", False, "STARTUP",
                     "lidar_quality" if scan.quality_reason else "confirmation", message, 0.0, 0.0, scan, {}, now, frame)
    return streak, stamp


def status_line(forward, yaw, frame, scan, now=None):
    """Four operator fields. Speeds are commands, not measured wheel motion."""
    now = time.monotonic() if now is None else now
    target = frame.target if camera_fault(frame, now) is None else None
    distance = (f"{target.z:.2f} m" if target is not None
                and math.isfinite(target.z) and 0.12 <= target.z <= 3.5 else "--")
    returns = "--"
    if lidar_fault(scan, now) is None:
        # Count actual positive returns only, never policy-inferred free space.
        count = sum(0 < mm < 65535 for mm in scan.raw_mm)
        returns = f"{count}/360"
    return (f"{'KJØRER' if forward or yaw else 'STOPP'} | Distanse {distance} | "
            f"Sving {yaw:+.2f} rad/s | Positive LiDAR-returer {returns}")


def bounded(low, high):
    def number(text):
        value = float(text)
        if not low <= value <= high:  
            raise argparse.ArgumentTypeError(f"ma vaere mellom {low:g} og {high:g}")
        return value
    return number


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add = parser.add_argument
    add("--version", action="version", version=f"marker_follower2 {VERSION}")
    add("--drive", action="store_true", help="Aktiver motorene. Uten flagget: kun avlesning.")
    add("--data", "--verbose", dest="verbose", action="store_true",
        help=argparse.SUPPRESS)
    add("--camera-only", action="store_true", help="Bruk bare kameraet. Ingen LiDAR eller hindringsstopp.")
    add("--assume-visible-obstacles", action="store_true",
        help="Nullregel: korte hull vurderes geometrisk; lengre antas frie ut til naermeste malte kant. "
             "Krever at hindringer gir lidarretur; ingen deteksjonsgaranti.")
    add("--min-obstacle-width-cm", type=bounded(0.5, 50), default=MIN_OBSTACLE_WIDTH_M*100,
        help="Minste projiserte objektbredde i LiDAR-planet. Standard: 5 cm.")
    add("--log-csv", type=Path, default=None, help="CSV-fil for beslutninger. Standard: ny datert fil i logs/.")
    add("--video", action="store_true",
        help="Lagre kameravideo med markering av gjenkjent markor. Samme navn som CSV-filen, med .avi.")
    add("--lidar-x-cm", type=bounded(-10, 10), default=-6.4,
        help="LiDAR relativt til hjulakselen, +forover, cm. Waffle standard -6.4.")
    add("--camera-left-cm", type=bounded(-50, 50), default=6.4,
        help="Kameralinsens forskyvning mot venstre i cm. Standard: 6.4.")
    add("--time", "--duration", dest="duration", type=bounded(1, 300), default=60.0,
        help="Kjoretid etter oppstart i sekunder, 1-300. Standard: 60.")
    add("--speed", "--max-speed", dest="max_speed", type=bounded(0.02, 0.16), default=0.16,
        help="Maks fremoverfart, 0.02-0.16 m/s. Standard: 0.16.")
    add("--distance", type=bounded(0.50, 2.00), default=TARGET_DISTANCE_M,
        help="Folgeavstand fra kameraet til markoren, 0.50-2.00 m. Standard: 0.60.")
    args = parser.parse_args()
    if args.assume_visible_obstacles and args.camera_only:
        parser.error("Nullregelen krever LiDAR og kan ikke brukes med --camera-only.")
    return args


def main():
    args = parse_args()
    use_lidar = not args.camera_only
    shutdown = threading.Event()
    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(number, lambda *_: shutdown.set())
    shared = Shared()
    policy = LidarPolicy(args.assume_visible_obstacles, args.min_obstacle_width_cm/100)
    geometry = RobotGeometry(lidar_x=args.lidar_x_cm/100)
    follower = Follower(geometry, args.max_speed, use_lidar, args.camera_left_cm/100, args.distance)
    robot = logger = scan = frame = video = None
    workers, causes = [], Counter()
    exit_reason, result = "stoppsignal mottatt", 0
    started = time.monotonic()
    try:
        os.environ["LIBCAMERA_LOG_LEVELS"] = "*:ERROR"
        logging.getLogger("picamera2").setLevel(logging.ERROR)
        print(status_line(0.0, 0.0, None, None), flush=True)
        matrix, distortion = load_calibration()
        cv2.setNumThreads(2)
        logger = ControlLog(args.log_csv, policy)
        video = VideoLog(Path(logger.path).with_suffix(".avi")) if args.video else None
        if args.drive:
            robot = OpenCR(OPENCR_DEVICE)
            robot.connect()
        if shutdown.is_set():
            return result
        jobs = [(camera_worker, (shared, shutdown, matrix, distortion, video))]
        if video is not None:
            jobs.append((video.run, (shutdown,)))
        if use_lidar:
            for forward, turn in ((True, 0), (False, 1), (False, -1), (True, 1), (True, -1)):
                swept_requirements(geometry, forward, turn)  # Compute the geometry before anything moves.
            jobs.append((lidar_worker, (shared, shutdown)))
        for job, arguments in jobs:
            workers.append(threading.Thread(target=job, args=arguments, daemon=True))
            workers[-1].start()
        started = time.monotonic()
        streak, stamp = 3, -math.inf
        if use_lidar:
            streak, stamp = startup_lidar(shared, shutdown, policy, logger, started, False)
        deadline = time.monotonic()+args.duration
        motors_on = False
        previous_forward = previous_yaw = 0.0
        last_tick, last_print, shown = time.monotonic(), -math.inf, ""
        while not shutdown.is_set():
            frame, scan, failure = shared.snapshot()
            now = time.monotonic()
            if failure is not None:
                raise RuntimeError(failure)
            if now >= deadline:
                exit_reason = f"tidsgrensen pa {args.duration:g} sekunder er nadd"
                break
            if use_lidar:
                scan = policy.prepare(scan)
                if scan is not None and scan.started_at > stamp:
                    stamp = scan.started_at
                    streak = 0 if scan.quality_reason else streak+1
            decision = follower.choose(frame, scan, now)
            if streak < 3:
                decision = hold("lidar_quality", "PAUSE: venter pa stabile LiDAR-data"
                                + ("; "+scan.quality_reason if scan is not None and scan.quality_reason else ""))
            checked = time.monotonic()
            if decision.forward or decision.yaw:
                faults = [fault for fault in (lidar_fault(scan, checked) if use_lidar else None,
                                              camera_fault(frame, checked)) if fault]
                if faults:
                    decision = hold(faults[0][0], "STOPP: data ble for gamle/ugyldige under beregningen: "
                                    + "; ".join(text for _, text in faults))
            dt = max(0.0, min(0.15, now-last_tick))
            last_tick = now
            forward = previous_forward = ramp(decision.forward, previous_forward, 0.20*dt)
            yaw = previous_yaw = ramp(decision.yaw, previous_yaw, 0.50*dt)
            if shutdown.is_set():
                break
            if robot is not None:
                if not motors_on and streak >= 3:
                    robot.enable()
                    motors_on = True
                    forward = yaw = previous_forward = previous_yaw = 0.0
                    decision = hold("confirmation", "OPPSTART: motorer aktivert; nullkommando denne sloyfen")
                if motors_on:
                    robot.set_velocity(forward, yaw)
            sent = time.monotonic()
            follower.record_command(sent, yaw if motors_on else 0.0, decision.searching)
            code = "motion" if forward or yaw else decision.code
            causes[code] += 1
            if video is not None:
                video.moving = bool(motors_on and (forward or yaw))
            report = follower.report
            logger.write(sent-started, "control", motors_on, "FOLLOW", code, decision.message,
                         forward, yaw, scan, report, checked, frame)
            # A dry run must not claim that the physical robot is moving.
            display_forward, display_yaw = (forward, yaw) if motors_on else (0.0, 0.0)
            line = status_line(display_forward, display_yaw, frame, scan)
            key = line.split(" |", 1)[0]
            if now-last_print >= 0.5 or key != shown:
                print(line, flush=True)
                shown, last_print = key, now
            shutdown.wait(min(0.08, max(0.0, deadline-time.monotonic())))
    except Exception as error:
        diagnostic_note(f"Kjoring avbrutt: {error}")
        exit_reason, result = f"feil: {error}", 1
    finally:
        if robot is not None:
            robot.close()
        if video is not None:
            video.moving = False
        print(status_line(0.0, 0.0, frame, scan), flush=True)
        shutdown.set()
        for worker in workers:
            worker.join(timeout=1.0)
        if logger is not None:
            try:
                logger.write(time.monotonic()-started, "shutdown", False, "STOPPED", "shutdown",
                             exit_reason, 0.0, 0.0, scan, {})
            finally:
                logger.close()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
