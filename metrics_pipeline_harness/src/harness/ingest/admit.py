"""Stages 1-2: admission and container inspection (plan §4.1, §4.2).

Runs in the parent process with the standard library only: nothing is extracted to disk and no XML
is parsed. Formats other than plain OOXML zips are routed, never guessed.
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

from harness.errors import ResourceLimitExceeded, UnsupportedWorkbook, WorkbookCorrupt
from harness.ingest.models import AdmissionResult, IngestLimits

ZIP_MAGIC = b"PK\x03\x04"
OLE_MAGIC = bytes.fromhex("D0CF11E0A1B11AE1")
SUPPORTED_EXTENSIONS = {".xlsx", ".xlsm"}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _vba_hash(archive: zipfile.ZipFile, infos: list[zipfile.ZipInfo], limits: IngestLimits) -> str | None:
    """SHA-256 of the VBA project bytes, recorded for review. The bytes are hashed, never interpreted."""
    for info in infos:
        if info.filename == "xl/vbaProject.bin":
            if info.file_size > limits.max_file_bytes or (
                info.compress_size and info.file_size / info.compress_size > limits.max_compression_ratio
            ):
                raise ResourceLimitExceeded("WB_ZIP_LIMIT", "vbaProject.bin over limits")
            return hashlib.sha256(archive.read(info)).hexdigest()
    return None


def admit(path: Path, limits: IngestLimits) -> AdmissionResult:
    if not path.is_file():
        raise UnsupportedWorkbook("WB_NOT_FOUND", str(path))
    size = path.stat().st_size
    if size > limits.max_file_bytes:
        raise ResourceLimitExceeded("WB_FILE_TOO_LARGE", f"{size} bytes > {limits.max_file_bytes}")
    with path.open("rb") as handle:
        head = handle.read(8)
    extension = path.suffix.lower()
    if head.startswith(OLE_MAGIC):
        # Legacy .xls and encrypted OOXML are both OLE compound files: route to an approved path.
        raise UnsupportedWorkbook(
            "WB_OLE_CONTAINER",
            "OLE compound file (.xls or encrypted OOXML); route to the approved parser or decryption path",
            evidence={"extension": extension},
        )
    if not head.startswith(ZIP_MAGIC):
        raise UnsupportedWorkbook("WB_NOT_SPREADSHEET", f"magic={head.hex()}", evidence={"extension": extension})
    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedWorkbook(
            "WB_EXTENSION_MISMATCH",
            f"zip container with extension {extension!r}; expected one of {sorted(SUPPORTED_EXTENSIONS)}",
        )
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            vba_sha256 = _vba_hash(archive, infos, limits)
    except zipfile.BadZipFile as exc:
        raise WorkbookCorrupt("WB_CORRUPT", str(exc)) from exc
    if len(infos) > limits.max_zip_entries:
        raise ResourceLimitExceeded("WB_ZIP_LIMIT", f"{len(infos)} entries > {limits.max_zip_entries}")
    uncompressed = sum(info.file_size for info in infos)
    if uncompressed > limits.max_uncompressed_bytes:
        raise ResourceLimitExceeded(
            "WB_ZIP_LIMIT", f"declared uncompressed size {uncompressed} > {limits.max_uncompressed_bytes}"
        )
    worst_ratio = 0.0
    for info in infos:
        if info.compress_size:
            ratio = info.file_size / info.compress_size
        elif info.file_size:
            ratio = float("inf")
        else:
            ratio = 0.0
        worst_ratio = max(worst_ratio, ratio)
        if ratio > limits.max_compression_ratio:
            raise ResourceLimitExceeded("WB_COMPRESSION_RATIO", f"{info.filename}: ratio {ratio:.0f}")
    names = {info.filename for info in infos}
    if "xl/workbook.xml" not in names and "xl/workbook.bin" not in names:
        raise UnsupportedWorkbook("WB_NOT_SPREADSHEET", "zip has no xl/workbook part")
    is_xlsb = "xl/workbook.bin" in names
    if is_xlsb:
        raise UnsupportedWorkbook("WB_XLSB", "binary workbook; route to the approved .xlsb path")
    has_vba = "xl/vbaProject.bin" in names
    return AdmissionResult(
        sha256=sha256_file(path),
        size_bytes=size,
        format="xlsm" if has_vba or extension == ".xlsm" else "xlsx",
        has_vba=has_vba,
        vba_sha256=vba_sha256,
        has_external_links=any(name.startswith("xl/externalLinks/") for name in names),
        has_connections="xl/connections.xml" in names,
        is_xlsb=is_xlsb,
        zip_entries=len(infos),
        uncompressed_bytes=uncompressed,
        max_compression_ratio=round(worst_ratio, 3),
    )
