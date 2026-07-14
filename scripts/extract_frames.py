#!/usr/bin/env python3
"""Extract image frames from a ROS2 rosbag (.db3 / sqlite3) without ROS.

ROS2 bags store ``sensor_msgs/msg/Image`` messages CDR-serialized in a sqlite
database. We deserialize the relevant fields (height, width, encoding, step,
data) with a tiny hand-written CDR reader (no rclpy / rosbag2 dependency) and
write uniformly-sampled frames as PNG/JPG.

This runs on the GPU server against a local copy of the bag (reading over a
network CIFS mount is too slow). Frames are written under a server-side data
directory that is never committed to the public repo (see scripts/check_no_leaks.py).

Usage:
  python3 scripts/extract_frames.py BAG.db3 --topic /sensing/camera/long \
      --num-frames 8 --out <server-side frames dir>
"""

from __future__ import annotations

import argparse
import sqlite3
import struct
from pathlib import Path
from typing import Iterator

from PIL import Image


# --- minimal CDR reader for the fields we need from sensor_msgs/msg/Image ----

class _CDR:
    """Read CDR little-endian primitives with 4-byte alignment (ROS2 default)."""

    def __init__(self, data: bytes, offset: int = 0):
        # Skip the 4-byte encapsulation header (ROS2 CDR open-fracture).
        self.data = data
        self.pos = offset

    def _align(self, size: int) -> None:
        self.pos = (self.pos + size - 1) // size * size

    def u32(self) -> int:
        self._align(4)
        value = struct.unpack_from("<I", self.data, self.pos)[0]
        self.pos += 4
        return value

    def boolean(self) -> bool:
        value = self.data[self.pos] != 0
        self.pos += 1
        return value

    def string(self) -> str:
        self._align(4)
        length = self.u32()
        value = self.data[self.pos : self.pos + max(0, length - 1)].decode(
            "utf-8", errors="replace"
        )
        self.pos += length
        return value

    def bytes(self) -> bytes:
        self._align(4)
        length = self.u32()
        value = self.data[self.pos : self.pos + length]
        self.pos += length
        return value


def _decode_image(payload: bytes) -> dict:
    """Decode a CDR sensor_msgs/msg/Image payload into raw pixel metadata."""
    cdr = _CDR(payload, offset=4)  # skip encapsulation header
    # header.stamp (sec: u32, nanosec: u32)
    cdr._align(4)
    stamp_sec = struct.unpack_from("<I", cdr.data, cdr.pos)[0]
    cdr.pos += 4
    stamp_nsec = struct.unpack_from("<I", cdr.data, cdr.pos)[0]
    cdr.pos += 4
    frame_id = cdr.string()
    height = cdr.u32()
    width = cdr.u32()
    encoding = cdr.string()
    cdr._align(4)
    step = cdr.u32()
    raw = cdr.bytes()
    return {
        "stamp_ns": stamp_sec * 1_000_000_000 + stamp_nsec,
        "frame_id": frame_id,
        "height": height,
        "width": width,
        "encoding": encoding,
        "step": step,
        "data": raw,
    }


_ENCODING_TO_MODE = {
    "rgb8": ("RGB", 3),
    "rgba8": ("RGBA", 4),
    "bgr8": ("RGB", 3),  # swap channels after load
    "bgra8": ("RGBA", 4),
    "8UC1": ("L", 1),
    "mono8": ("L", 1),
}


def _to_pil(msg: dict) -> Image.Image:
    height, width, encoding = msg["height"], msg["width"], msg["encoding"]
    if encoding not in _ENCODING_TO_MODE:
        raise ValueError(f"unsupported image encoding: {encoding!r}")
    mode, channels = _ENCODING_TO_MODE[encoding]
    expected = msg["step"] * height
    if len(msg["data"]) < expected:
        raise ValueError(
            f"image data too short: have {len(msg['data'])}, need {expected}"
        )
    image = Image.frombytes(mode, (width, height), msg["data"][:expected])
    if encoding in {"bgr8", "bgra8"}:
        # swap R and B channels
        bands = list(image.split())
        bands[0], bands[2] = bands[2], bands[0]
        image = Image.merge(mode, bands)
    return image


def iter_messages(db_path: str, topic: str) -> Iterator[dict]:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("SELECT id FROM topics WHERE name = ?", (topic,))
    row = cur.fetchone()
    if row is None:
        # Show available topics to help the user.
        avail = [r[0] for r in cur.execute("SELECT name FROM topics")]
        con.close()
        raise KeyError(f"topic {topic!r} not found; available: {avail}")
    topic_id = row[0]
    for (payload,) in cur.execute(
        "SELECT data FROM messages WHERE topic_id = ? ORDER BY timestamp",
        (topic_id,),
    ):
        if payload is None:
            continue
        try:
            yield _decode_image(payload)
        except Exception:
            continue
    con.close()


def extract(db_path: str, topic: str, num_frames: int, out_dir: Path, fmt: str) -> list[Path]:
    messages = list(iter_messages(db_path, topic))
    if not messages:
        raise RuntimeError(f"no decodable image messages on topic {topic!r}")
    total = len(messages)
    # Uniform sampling like the video_understanding cookbook (np.linspace).
    if num_frames >= total:
        indices = list(range(total))
    else:
        step = total / num_frames
        indices = [min(total - 1, int(i * step)) for i in range(num_frames)]
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for rank, idx in enumerate(indices, start=1):
        image = _to_pil(messages[idx])
        path = out_dir / f"frame_{rank:03d}_of_{len(indices):03d}.{fmt}"
        image.save(path)
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bag", help="path to the .db3 sqlite file")
    parser.add_argument("--topic", required=True, help="image topic, e.g. /sensing/camera/long")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--out", required=True, help="output directory (server-side)")
    parser.add_argument("--format", default="jpg", choices=("jpg", "png"))
    parser.add_argument("--list-topics", action="store_true", help="print image topics and exit")
    args = parser.parse_args(argv)

    if args.list_topics:
        con = sqlite3.connect(args.bag)
        for name, mtype in con.execute("SELECT name, type FROM topics"):
            if "Image" in mtype:
                print(name)
        con.close()
        return 0

    written = extract(args.bag, args.topic, args.num_frames, Path(args.out), args.format)
    print(f"extracted {len(written)} frames from {args.topic!r} -> {args.out}")
    for path in written:
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
