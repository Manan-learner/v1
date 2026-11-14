"""Multi-LLM non-destructive WAL-safe backup & status verifier (Google Drive Desktop folder + rclone)."""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from v1_memory.index_store import default_state_home, rebuild_portable_index

DEFAULT_REMOTE = os.environ.get("V1_GDRIVE_REMOTE", os.environ.get("AGY_GDRIVE_REMOTE", "gdrive:AI-Agent-Memory"))
EXCLUDE_DIRS = {"node_modules", ".git", "__pycache__", ".venv", "cache"}
EXCLUDE_RUNTIME_FILES = {"cloud_sync_state.json", "cloud_sync.log", "context_finder_seen.json"}
COMMON_DRIVE_DIRS = [
    Path(r"G:\My Drive\AI-Agent-Memory"),
    Path(r"H:\My Drive\AI-Agent-Memory"),
    Path.home() / "Google Drive" / "My Drive" / "AI-Agent-Memory",
    Path.home() / "Google Drive" / "AI-Agent-Memory",
]


def load_sync_config(state_home: Path) -> dict:
    cfg = state_home / "cloud_sync_config.json"
    if cfg.exists():
        try:
            return json.loads(cfg.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def detect_drive_folder(state_home: Path, explicit_dir: str | None) -> Path | None:
    if explicit_dir:
        return Path(explicit_dir).expanduser()
    env_dir = os.environ.get("V1_GDRIVE_DIR") or os.environ.get("AGY_GDRIVE_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    cfg = load_sync_config(state_home)
    if cfg.get("target_dir"):
        return Path(cfg["target_dir"]).expanduser()
    for cand in COMMON_DRIVE_DIRS:
        if cand.parent.exists():
            return cand
    return None


def find_rclone() -> str | None:
    found = shutil.which("rclone")
    if found:
        return found
    if os.name == "nt":
        for pat in [
            os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Rclone.Rclone*\*\rclone.exe"),
            os.path.expandvars(r"%ProgramFiles%\rclone\rclone.exe"),
            os.path.expandvars(r"%USERPROFILE%\scoop\shims\rclone.exe"),
        ]:
            matches = sorted(glob.glob(pat))
            if matches:
                return matches[-1]
    return None


def probe_rclone(rclone_exe: str | None, remote_name: str) -> tuple[bool, bool, str | None]:
    if not rclone_exe:
        return False, False, "rclone executable not found"
    try:
        res = subprocess.run([rclone_exe, "listremotes"], capture_output=True, text=True, check=False)
    except OSError as exc:
        return False, False, f"blocked or unrunnable ({exc})"
    if res.returncode != 0:
        return False, False, res.stderr.strip() or f"exit code {res.returncode}"
    remotes = [line.strip().rstrip(":") for line in res.stdout.splitlines() if line.strip()]
    return True, (remote_name in remotes), None


def sqlite_safe_snapshot(src_db: Path, dst_db: Path) -> None:
    """Create a transactionally consistent copy even while WAL writers are active."""
    dst_db.parent.mkdir(parents=True, exist_ok=True)
    if dst_db.exists():
        dst_db.unlink()
    src_conn = sqlite3.connect(f"file:{src_db}?mode=ro", uri=True, timeout=15.0)
    try:
        dst_conn = sqlite3.connect(str(dst_db))
        try:
            src_conn.backup(dst_conn)
        finally:
            dst_conn.close()
    finally:
        src_conn.close()
    wal = Path(str(src_db) + "-wal")
    latest = max(src_db.stat().st_mtime, wal.stat().st_mtime) if wal.exists() else src_db.stat().st_mtime
    os.utime(dst_db, (latest, latest))


def discover_sync_sources(home: Path, state_home: Path, providers: list[str] | None, extra_dirs: list[str] | None) -> dict[str, Path]:
    """Return {remote_subpath: local_dir} across requested AI coding CLIs."""
    selected = set(providers or ["antigravity", "claude", "codex", "v1-memory"])
    candidates: dict[str, Path] = {}
    if "antigravity" in selected:
        p = home / ".gemini"
        if p.exists():
            candidates["providers/antigravity"] = p
    if "claude" in selected:
        p = home / ".claude"
        if p.exists():
            candidates["providers/claude"] = p
    if "codex" in selected:
        p = home / ".codex"
        if p.exists():
            candidates["providers/codex"] = p
    if "v1-memory" in selected and state_home.exists():
        candidates["providers/v1-memory"] = state_home
    for spec in extra_dirs or []:
        if "=" in spec:
            name, path_str = spec.split("=", 1)
        else:
            p = Path(spec).expanduser()
            name, path_str = p.name, spec
        target = Path(path_str).expanduser()
        if target.exists():
            candidates[f"providers/custom/{name}"] = target
    return candidates


def collect_fingerprint(sources: dict[str, Path]) -> dict[str, list[int]]:
    fp: dict[str, list[int]] = {}
    for prefix, root in sorted(sources.items()):
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE_DIRS)
            for fname in sorted(filenames):
                if fname in EXCLUDE_RUNTIME_FILES or fname.endswith((".db-shm", ".tmp")):
                    continue
                fpath = Path(dirpath) / fname
                try:
                    st = fpath.stat()
                except OSError:
                    continue
                fp[f"{prefix}/{fpath.relative_to(root).as_posix()}"] = [st.st_size, st.st_mtime_ns]
    return fp


def diff_fingerprints(old: dict[str, list[int]], new: dict[str, list[int]]) -> tuple[int, int, int]:
    added = sum(1 for k in new if k not in old)
    modified = sum(1 for k, v in new.items() if k in old and old[k] != v)
    removed = sum(1 for k in old if k not in new)
    return added, modified, removed


def folder_copy_update(src: Path, dst: Path, exclude_names: set[str] | None = None, dry_run: bool = False) -> int:
    copied = 0
    if not src.exists():
        return copied
    ex = exclude_names or set()
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        rel = Path(dirpath).relative_to(src)
        for fname in filenames:
            if fname in ex or fname in EXCLUDE_RUNTIME_FILES or fname.endswith((".db-wal", ".db-shm", ".tmp")):
                continue
            sfile = Path(dirpath) / fname
            dfile = dst / rel / fname
            try:
                sst = sfile.stat()
            except OSError:
                continue
            needs_copy = True
            if dfile.exists():
                dst_st = dfile.stat()
                if int(sst.st_mtime) <= int(dst_st.st_mtime) and sst.st_size == dst_st.st_size:
                    needs_copy = False
            if needs_copy:
                if not dry_run:
                    dfile.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(sfile, dfile)
                copied += 1
    return copied


def stage_wal_snapshots(sources: dict[str, Path], stage_root: Path) -> tuple[set[str], int]:
    """Snapshot every active SQLite DB cleanly into stage_root matching remote subpaths."""
    snapshotted_rel_paths: set[str] = set()
    wal_count = 0
    for remote_sub, root in sources.items():
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
            for fname in filenames:
                if fname.endswith((".db", ".sqlite")) or fname.endswith((".db-wal", ".sqlite-wal")):
                    if fname.endswith("-wal"):
                        wal_count += 1
                        base_name = fname[:-4]
                        src_db = Path(dirpath) / base_name
                    else:
                        src_db = Path(dirpath) / fname
                        if not Path(str(src_db) + "-wal").exists():
                            continue
                    if src_db.exists():
                        rel = src_db.relative_to(root)
                        dst_db = stage_root / remote_sub / rel
                        if str(dst_db) not in snapshotted_rel_paths:
                            sqlite_safe_snapshot(src_db, dst_db)
                            snapshotted_rel_paths.add(src_db.name)
    return snapshotted_rel_paths, wal_count


def run_sync(
    target_dir: str | None = None,
    remote: str = DEFAULT_REMOTE,
    providers: list[str] | None = None,
    extra_dirs: list[str] | None = None,
    dry_run: bool = False,
    force: bool = False,
    reindex: bool = True,
    home: Path | None = None,
    state_home: Path | None = None,
) -> dict:
    user_home = home or Path.home()
    resolved_state = default_state_home(state_home)
    resolved_state.mkdir(parents=True, exist_ok=True)

    if reindex and not dry_run:
        rebuild_portable_index(resolved_state / "index.db", home=user_home)

    sources = discover_sync_sources(user_home, resolved_state, providers, extra_dirs)
    new_fp = collect_fingerprint(sources)
    state_file = resolved_state / "cloud_sync_state.json"
    old_fp: dict[str, list[int]] = {}
    last_sync_iso = None
    if state_file.exists():
        try:
            saved = json.loads(state_file.read_text(encoding="utf-8"))
            old_fp = saved.get("files", {})
            last_sync_iso = saved.get("last_sync_utc")
        except Exception:
            old_fp = {}

    added, modified, removed = diff_fingerprints(old_fp, new_fp)
    if (added + modified + removed) == 0 and not force:
        return {
            "status": "skipped",
            "reason": f"No changes detected since {last_sync_iso or 'last sync'}",
            "tracked_files": len(new_fp),
            "providers": sorted(sources.keys()),
        }

    drive_folder = detect_drive_folder(resolved_state, target_dir)
    rclone_exe = find_rclone()
    remote_name = remote.split(":", 1)[0]
    rclone_allowed, remote_ok, rclone_err = probe_rclone(rclone_exe, remote_name)
    hostname = socket.gethostname()
    now_utc = datetime.now(timezone.utc).isoformat()

    with tempfile.TemporaryDirectory(prefix="v1_sync_") as tmpdir_str:
        stage = Path(tmpdir_str)
        snap_names, wal_count = stage_wal_snapshots(sources, stage)
        manifest = {
            "hostname": hostname,
            "platform": sys.platform,
            "synced_at_utc": now_utc,
            "providers": sorted(sources.keys()),
            "wal_snapshots": wal_count,
            "tracked_files": len(new_fp),
            "diff": {"added": added, "modified": modified, "removed": removed},
        }
        mpath = stage / "manifests" / f"{hostname}_manifest.json"
        mpath.parent.mkdir(parents=True, exist_ok=True)
        mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        if drive_folder is not None:
            copied = 0
            for remote_sub, local_root in sources.items():
                copied += folder_copy_update(local_root, drive_folder / remote_sub, exclude_names=snap_names, dry_run=dry_run)
            copied += folder_copy_update(stage, drive_folder, dry_run=dry_run)
            if not dry_run:
                state_file.write_text(
                    json.dumps(
                        {
                            "last_sync_utc": now_utc,
                            "hostname": hostname,
                            "backend": "drive_folder",
                            "destination": str(drive_folder),
                            "files": new_fp,
                        }
                    ),
                    encoding="utf-8",
                )
            return {
                "status": "dry_run" if dry_run else "ok",
                "backend": "drive_folder",
                "destination": str(drive_folder),
                "copied_files": copied,
                "diff": manifest["diff"],
                "providers": manifest["providers"],
            }

        if not rclone_allowed or not remote_ok:
            return {
                "status": "pending_target",
                "reason": rclone_err or f"rclone remote '{remote_name}:' not configured",
                "diff": manifest["diff"],
            }

        for remote_sub, local_root in sources.items():
            cmd = [rclone_exe, "copy", str(local_root), f"{remote}/{remote_sub}", "--update", "--stats=0", "--log-level=ERROR"]
            for ex in ["*.db-wal", "*.db-shm", "*.tmp"] + sorted(snap_names):
                cmd.extend(["--exclude", ex])
            if dry_run:
                cmd.append("--dry-run")
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode != 0:
                raise RuntimeError(res.stderr.strip() or res.stdout.strip())

        stage_cmd = [rclone_exe, "copy", str(stage), remote, "--update", "--stats=0", "--log-level=ERROR"]
        if dry_run:
            stage_cmd.append("--dry-run")
        res2 = subprocess.run(stage_cmd, capture_output=True, text=True, check=False)
        if res2.returncode != 0:
            raise RuntimeError(res2.stderr.strip() or res2.stdout.strip())

        if not dry_run:
            state_file.write_text(
                json.dumps(
                    {
                        "last_sync_utc": now_utc,
                        "hostname": hostname,
                        "backend": "rclone",
                        "destination": remote,
                        "files": new_fp,
                    }
                ),
                encoding="utf-8",
            )
        return {
            "status": "dry_run" if dry_run else "ok",
            "backend": "rclone",
            "destination": remote,
            "diff": manifest["diff"],
            "providers": manifest["providers"],
        }


def get_sync_status(
    target_dir: str | None = None,
    remote: str = DEFAULT_REMOTE,
    providers: list[str] | None = None,
    extra_dirs: list[str] | None = None,
    home: Path | None = None,
    state_home: Path | None = None,
) -> dict:
    user_home = home or Path.home()
    resolved_state = default_state_home(state_home)
    sources = discover_sync_sources(user_home, resolved_state, providers, extra_dirs)
    fp = collect_fingerprint(sources)
    state_file = resolved_state / "cloud_sync_state.json"
    saved = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else {}
    added, modified, removed = diff_fingerprints(saved.get("files", {}), fp)
    drive_folder = detect_drive_folder(resolved_state, target_dir)
    rclone_exe = find_rclone()
    rclone_allowed, rclone_auth, rclone_err = probe_rclone(rclone_exe, remote.split(":", 1)[0])
    return {
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "active_backend": "drive_folder" if drive_folder else ("rclone" if (rclone_allowed and rclone_auth) else "not_connected"),
        "detected_drive_folder": str(drive_folder) if drive_folder else None,
        "rclone": {"found": bool(rclone_exe), "allowed": rclone_allowed, "authenticated": rclone_auth, "note": rclone_err},
        "providers": sorted(sources.keys()),
        "tracked_files": len(fp),
        "last_sync_utc": saved.get("last_sync_utc"),
        "unsynced_changes": {"added": added, "modified": modified, "removed": removed},
    }


def main_sync() -> int:
    parser = argparse.ArgumentParser(description="Incremental non-destructive backup for AI CLI conversations and skills.")
    parser.add_argument("--target-dir", help="Mounted Drive folder (e.g. G:\\My Drive\\AI-Agent-Memory)")
    parser.add_argument("--remote", default=DEFAULT_REMOTE, help=f"rclone remote (default: {DEFAULT_REMOTE})")
    parser.add_argument("--provider", action="append", help="antigravity, claude, codex, v1-memory (repeatable)")
    parser.add_argument("--extra-dir", action="append", help="Extra folder name=path or path to back up")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-reindex", action="store_true")
    args = parser.parse_args()

    res = run_sync(
        target_dir=args.target_dir,
        remote=args.remote,
        providers=args.provider,
        extra_dirs=args.extra_dir,
        dry_run=args.dry_run,
        force=args.force,
        reindex=not args.no_reindex,
    )
    print(json.dumps(res, indent=2))
    return 0 if res["status"] in ("ok", "skipped", "dry_run") else 3


def main_status() -> int:
    parser = argparse.ArgumentParser(description="Check AI Agent Cloud Memory backup status.")
    parser.add_argument("--target-dir")
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    parser.add_argument("--provider", action="append")
    parser.add_argument("--extra-dir", action="append")
    args = parser.parse_args()
    print(json.dumps(get_sync_status(args.target_dir, args.remote, args.provider, args.extra_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main_sync())
