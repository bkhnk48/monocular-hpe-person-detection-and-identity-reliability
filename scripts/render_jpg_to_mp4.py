import os
import re
import argparse
import shutil
import subprocess
import tempfile
import time
import json
from datetime import datetime, timedelta
from tqdm import tqdm
import threading


class NvidiaGpuMonitor:
    """Theo dõi tài nguyên GPU NVIDIA trong lúc render."""

    def __init__(self, pbar, interval=2):
        self.pbar = pbar
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = None
        self.available = None
        self.last_status = "GPU: đang kiểm tra..."

    def read_gpu_stats(self):
        """Đọc chỉ số GPU bằng nvidia-smi."""

        query = [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,"
            "utilization.memory,memory.used,memory.total,"
            "temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]

        try:
            result = subprocess.run(
                query,
                capture_output=True,
                text=True,
                timeout=3,
                check=True,
            )

            gpu_lines = [
                line.strip()
                for line in result.stdout.splitlines()
                if line.strip()
            ]

            if not gpu_lines:
                return None

            # Encoder/decoder utilization được truy vấn riêng,
            # vì không phải GPU nào cũng hỗ trợ các chỉ số này.
            encoder_decoder = []

            try:
                ed_result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.encoder,"
                        "utilization.decoder",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    check=True,
                )

                encoder_decoder = [
                    line.strip().split(",")
                    for line in ed_result.stdout.splitlines()
                    if line.strip()
                ]

            except (subprocess.SubprocessError, OSError):
                pass

            statuses = []

            for index, line in enumerate(gpu_lines):
                fields = [x.strip() for x in line.split(",")]

                if len(fields) < 7:
                    continue

                name, gpu, mem_util, mem_used, mem_total, temp, power = fields[:7]

                status = (
                    f"GPU{index} {name}: "
                    f"Core={gpu}% | "
                    f"MemUtil={mem_util}% | "
                    f"VRAM={mem_used}/{mem_total}MiB | "
                    f"Temp={temp}C | Power={power}W"
                )

                if index < len(encoder_decoder):
                    ed = encoder_decoder[index]

                    if len(ed) >= 2:
                        status += (
                            f" | ENC={ed[0].strip()}%"
                            f" DEC={ed[1].strip()}%"
                        )

                statuses.append(status)

            return " || ".join(statuses) if statuses else None

        except (subprocess.SubprocessError, OSError):
            return None

    def monitor_loop(self):
        while not self.stop_event.is_set():
            stats = self.read_gpu_stats()

            if stats:
                self.available = True
                self.last_status = stats
            else:
                self.available = False
                self.last_status = (
                    "GPU NVIDIA: không đọc được nvidia-smi"
                )

            # Hiển thị trực tiếp trên thanh tqdm
            try:
                self.pbar.set_postfix_str(
                    self.last_status,
                    refresh=True,
                )
            except Exception:
                pass

            self.stop_event.wait(self.interval)

    def start(self):
        self.thread = threading.Thread(
            target=self.monitor_loop,
            daemon=True,
        )
        self.thread.start()

    def stop(self):
        self.stop_event.set()

        if self.thread is not None:
            self.thread.join(timeout=5)

VIDEO_PREFIX = "harmony4D"
DEFAULT_FPS = 20

# Tên frame: <prefix><số><.jpg|.jpeg>, không phân biệt hoa/thường.
FRAME_NAME_RE = re.compile(r"^(.*?)(\d+)(\.jpe?g)$", flags=re.IGNORECASE)


def find_jpg_sequence(images_dir):
    """
    Tìm chuỗi JPG được đánh số liên tiếp.

    Hỗ trợ số frame ở cuối tên có độ rộng khác nhau trong cùng một thư mục,
    ví dụ 5 chữ số (00000.jpg ... 99999.jpg) lẫn 6 chữ số (100000.jpg,
    000123.jpg ...). Thứ tự frame được xác định theo GIÁ TRỊ số, không theo
    thứ tự chữ cái.

    Các frame phải có cùng prefix, cùng đuôi file và không thiếu/trùng index.

    Trả về dict:
      - input_pattern, start_number: pattern FFmpeg dùng trực tiếp được
        (None nếu độ rộng số không đồng nhất -> cần "staging" khi render)
      - frame_paths: danh sách đường dẫn frame theo đúng thứ tự
      - jpg_count, widths
    """
    parsed = []
    for name in os.listdir(images_dir):
        match = FRAME_NAME_RE.match(name)
        if not match:
            continue
        prefix, number, suffix = match.groups()
        parsed.append((int(number), len(number), prefix, suffix, name))

    if not parsed:
        raise ValueError(f"Không tìm thấy file JPG có số ở cuối tên trong: {images_dir}")

    prefixes = {item[2] for item in parsed}
    suffixes = {item[3] for item in parsed}
    if len(prefixes) != 1 or len(suffixes) != 1:
        raise ValueError(
            "Các file JPG phải có cùng prefix và phần mở rộng. "
            f"Prefix: {sorted(prefixes)[:5]} | đuôi: {sorted(suffixes)}"
        )

    parsed.sort(key=lambda item: item[0])
    numbers = [item[0] for item in parsed]

    if len(numbers) != len(set(numbers)):
        seen, dups = set(), []
        for n, _, _, _, name in parsed:
            if n in seen:
                dups.append(name)
            seen.add(n)
        raise ValueError(
            "Có frame bị trùng số thứ tự (ví dụ 01234.jpg và 001234.jpg): "
            + ", ".join(dups[:5])
        )

    expected = range(numbers[0], numbers[0] + len(numbers))
    if numbers != list(expected):
        missing = sorted(set(range(numbers[0], numbers[-1] + 1)) - set(numbers))
        preview = ", ".join(map(str, missing[:10]))
        raise ValueError(f"Chuỗi frame bị thiếu index: {preview}")

    widths = sorted({item[1] for item in parsed})
    prefix = next(iter(prefixes))
    suffix = next(iter(suffixes))
    frame_paths = [os.path.join(images_dir, item[4]) for item in parsed]

    # Pattern %0Nd chỉ dùng được khi mọi tên có cùng độ rộng số
    # và đường dẫn không chứa ký tự '%' (FFmpeg hiểu '%' là định dạng).
    input_pattern = None
    if len(widths) == 1 and "%" not in images_dir and "%" not in prefix + suffix:
        input_pattern = os.path.join(images_dir, f"{prefix}%0{widths[0]}d{suffix}")

    return {
        "input_pattern": input_pattern,
        "start_number": numbers[0],
        "frame_paths": frame_paths,
        "jpg_count": len(numbers),
        "widths": widths,
    }


def stage_frames(frame_paths, staging_dir):
    """
    Tạo bản đánh số lại 000000.jpg, 000001.jpg, ... trong staging_dir,
    ưu tiên symlink -> hardlink -> copy. Trả về pattern FFmpeg.
    """
    width = max(6, len(str(len(frame_paths) - 1)))
    for index, src in enumerate(frame_paths):
        dst = os.path.join(staging_dir, f"{index:0{width}d}.jpg")
        try:
            os.symlink(os.path.abspath(src), dst)
        except OSError:
            try:
                os.link(src, dst)
            except OSError:
                shutil.copyfile(src, dst)
    return os.path.join(staging_dir, f"%0{width}d.jpg")


def make_task(images_dir, output_filepath, sequence, **extra):
    task = {
        "images_dir": images_dir,
        "output_filepath": output_filepath,
        "jpg_count": sequence["jpg_count"],
        "input_pattern": sequence["input_pattern"],
        "start_number": sequence["start_number"],
        "frame_paths": sequence["frame_paths"],
        "widths": sequence["widths"],
    }
    task.update(extra)
    return task


def find_exo_tasks(root_dir):
    """Tìm exo ở mọi cấp và lấy các thư mục camX/images có JPG."""
    tasks = []
    print("Đang quét đệ quy cấu trúc thư mục...")

    for current_path, dirnames, _ in os.walk(root_dir):
        if os.path.basename(current_path) != "exo":
            continue

        exo_path = current_path
        c_name = os.path.basename(os.path.dirname(exo_path))

        for cam_name in os.listdir(exo_path):
            if not (
                cam_name.startswith("cam")
                and cam_name[3:].isdigit()
                and int(cam_name[3:]) > 0
            ):
                continue

            cam_path = os.path.join(exo_path, cam_name)
            if not os.path.isdir(cam_path):
                continue

            images_path = os.path.join(cam_path, "images")
            if not os.path.isdir(images_path):
                continue

            try:
                sequence = find_jpg_sequence(images_path)
            except ValueError as exc:
                print(f"[Bỏ qua] {images_path}: {exc}")
                continue

            output_filename = (
                f"{VIDEO_PREFIX}_{c_name}_{cam_name}_ffmpeg.mp4"
            )

            tasks.append(make_task(
                images_path,
                os.path.join(images_path, output_filename),
                sequence,
                c_name=c_name,
                cam_name=cam_name,
            ))

        # Không quét lặp thư mục con của exo
        dirnames.clear()

    return tasks


def validate_video(filepath, expected_frames=None):
    """
    Xác thực MP4 bằng FFprobe.
    Trả về (True, thông tin) nếu video đạt các kiểm tra cơ bản
    và đọc được các gói dữ liệu hình ảnh. Nếu có expected_frames thì
    số packet phải bằng đúng số frame JPG.
    """

    if not os.path.isfile(filepath):
        return False, "File không tồn tại"

    if os.path.getsize(filepath) == 0:
        return False, "File rỗng"

    # Bước 1: Kiểm tra metadata, stream hình ảnh và thời lượng
    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries",
        "format=duration:stream=codec_type,codec_name,width,height",
        "-of", "json",
        filepath,
    ]

    try:
        result = subprocess.run(
            probe_cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )

        metadata = json.loads(result.stdout)
        streams = metadata.get("streams", [])

        video_streams = [
            stream for stream in streams
            if stream.get("codec_type") == "video"
        ]

        if not video_streams:
            return False, "Không có luồng hình ảnh"

        # Kiểm tra kích thước hình ảnh
        if not any(
            stream.get("width", 0) > 0
            and stream.get("height", 0) > 0
            for stream in video_streams
        ):
            return False, "Kích thước hình ảnh không hợp lệ"

        duration = float(
            metadata.get("format", {}).get("duration", 0)
        )

        if duration <= 0:
            return False, "Thời lượng video không hợp lệ"

    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        json.JSONDecodeError,
        ValueError,
        OSError,
    ) as exc:
        return False, f"FFprobe không đọc được metadata: {exc}"

    # Bước 2: Đọc/kiểm tra các gói dữ liệu của toàn bộ file.
    # -count_packets chỉ đếm packet, không giải mã toàn bộ hình ảnh.
    check_cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-count_packets",
        "-show_entries", "stream=nb_read_packets",
        "-of", "json",
        filepath,
    ]

    try:
        result = subprocess.run(
            check_cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=600,
        )

        packet_info = json.loads(result.stdout)
        packet_streams = packet_info.get("streams", [])

        if not packet_streams:
            return False, "Không đọc được packet hình ảnh"

        packet_count = int(
            packet_streams[0].get("nb_read_packets", 0)
        )

        if packet_count <= 0:
            return False, "Không có packet hình ảnh"

        if expected_frames is not None and packet_count != expected_frames:
            return False, (
                f"Số frame trong video ({packet_count}) khác số JPG "
                f"({expected_frames})"
            )

        return True, {
            "duration": duration,
            "packet_count": packet_count,
            "codec": video_streams[0].get("codec_name", "unknown"),
        }

    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        json.JSONDecodeError,
        ValueError,
        OSError,
    ) as exc:
        return False, f"Lỗi khi kiểm tra packet: {exc}"


def render_video(task, fps):
    """Render ra file tạm, xác thực rồi mới thay file đích."""

    output_path = task["output_filepath"]
    temp_path = output_path[:-4] + ".tmp.mp4"
    staging = None

    try:
        # Dọn file tạm còn sót lại từ lần chạy trước
        if os.path.exists(temp_path):
            os.remove(temp_path)

        if task["input_pattern"]:
            input_pattern = task["input_pattern"]
            start_number = task["start_number"]
        else:
            # Tên frame lẫn 5 và 6 chữ số: đánh số lại qua thư mục tạm
            staging = tempfile.TemporaryDirectory(prefix="render_frames_")
            input_pattern = stage_frames(task["frame_paths"], staging.name)
            start_number = 0

        cmd = [
            "ffmpeg",
            "-y",
            "-framerate", str(fps),
            "-start_number", str(start_number),
            "-i", input_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-v", "error",
            temp_path,
        ]

        subprocess.run(cmd, check=True)

        valid, info = validate_video(temp_path, task["jpg_count"])

        if not valid:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False, f"Video render ra không hợp lệ: {info}"

        # Chỉ thay file đích sau khi file tạm đã hợp lệ
        os.replace(temp_path, output_path)
        return True, info

    except subprocess.CalledProcessError as exc:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False, f"FFmpeg thất bại: {exc}"

    except FileNotFoundError as exc:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False, f"Không tìm thấy chương trình: {exc}"

    except OSError as exc:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False, f"Lỗi file: {exc}"

    finally:
        if staging is not None:
            staging.cleanup()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render các frame JPG thành MP4. Có thể render một thư mục cụ thể "
            "(--images) hoặc quét cấu trúc Harmony4D exo/camX/images như trước."
        )
    )
    parser.add_argument(
        "--images",
        type=str,
        default=None,
        help="Đường dẫn thư mục chứa các frame JPG cần render.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help=f"FPS video đầu ra (mặc định: {DEFAULT_FPS}).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Đường dẫn MP4 đầu ra; chỉ dùng cùng --images. "
            "Mặc định: <thư_mục_cha>/<tên_thư_mục_cha>.mp4."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.fps <= 0:
        raise SystemExit("Lỗi: --fps phải lớn hơn 0.")

    if args.output and not args.images:
        raise SystemExit("Lỗi: --output chỉ được sử dụng cùng với --images.")

    if args.images:
        images_dir = os.path.abspath(os.path.expanduser(args.images))
        if not os.path.isdir(images_dir):
            raise SystemExit(f"Lỗi: thư mục --images không tồn tại: {images_dir}")

        try:
            sequence = find_jpg_sequence(images_dir)
        except ValueError as exc:
            raise SystemExit(f"Lỗi dữ liệu ảnh: {exc}")

        if args.output:
            output_path = os.path.abspath(os.path.expanduser(args.output))
        else:
            parent_name = os.path.basename(os.path.dirname(images_dir)) or "video"
            output_path = os.path.join(
                os.path.dirname(images_dir), f"{parent_name}.mp4"
            )

        if not output_path.lower().endswith(".mp4"):
            raise SystemExit("Lỗi: --output phải có phần mở rộng .mp4.")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        tasks = [make_task(images_dir, output_path, sequence)]
        mode_description = f"Render một thư mục: {images_dir}"
    else:
        root_dir = os.getcwd()
        tasks = find_exo_tasks(root_dir)
        mode_description = f"Quét cấu trúc Harmony4D từ: {root_dir}"

    print(f"\nChế độ: {mode_description}")
    print(f"FPS đầu ra: {args.fps:g}")
    print(f"Tìm thấy {len(tasks)} task render.")

    mixed = [t for t in tasks if len(t["widths"]) > 1]
    if mixed:
        print(
            f"Có {len(mixed)} thư mục có tên frame lẫn nhiều độ rộng số "
            f"(ví dụ {mixed[0]['widths']} chữ số) -> sắp xếp theo giá trị số."
        )

    if not tasks:
        print("Không có dữ liệu cần xử lý.")
        return

    # Xác thực video đã có từ trước
    pending_tasks = []
    skipped_videos = []
    invalid_existing = []

    print("Đang kiểm tra các video MP4 đã tồn tại...")

    for task in tqdm(tasks, desc="Kiểm tra video", unit="video"):
        path = task["output_filepath"]

        if not os.path.exists(path):
            pending_tasks.append(task)
            continue

        valid, info = validate_video(path, task["jpg_count"])

        if valid:
            skipped_videos.append({"path": path, "info": info})
        else:
            invalid_existing.append({"path": path, "reason": info})
            pending_tasks.append(task)

    print("\n========== VIDEO HỢP LỆ ĐÃ CÓ ==========")
    for index, video in enumerate(skipped_videos, start=1):
        print(
            f"{index}. {video['path']} | "
            f"{video['info']['duration']:.2f}s | "
            f"{video['info']['codec']}"
        )

    if invalid_existing:
        print("\n========== VIDEO CŨ KHÔNG HỢP LỆ ==========")
        for item in invalid_existing:
            print(f"- {item['path']}\n  Lý do: {item['reason']}")
        print("Các video này sẽ được render lại.")

    if not pending_tasks:
        print("\nTất cả video đều hợp lệ. Không cần render.")
        print(f"Số video được bỏ qua: {len(skipped_videos)}")
        return

    print(f"\nSố video cần render: {len(pending_tasks)}")

    start_time = time.time()
    created_videos = []
    failed_videos = []
    ffmpeg_missing = False

    with tqdm(
        total=len(pending_tasks),
        desc="Đang render",
        unit="video",
        bar_format=(
            "{l_bar}{bar}| {n_fmt}/{total_fmt} "
            "[Elapsed: {elapsed} | Remaining: {remaining}]"
        ),
    ) as pbar:
        gpu_monitor = NvidiaGpuMonitor(pbar, interval=2)
        gpu_monitor.start()

        try:
            for task in pending_tasks:
                task_start = time.time()
                path = task["output_filepath"]

                try:
                    success, info = render_video(task, args.fps)

                    if success:
                        created_videos.append({
                            "path": path,
                            "duration_render": time.time() - task_start,
                            "duration_video": info["duration"],
                            "packets": info["packet_count"],
                        })
                    else:
                        failed_videos.append({"path": path, "reason": info})
                        print(f"\n[Lỗi] {path}\n  {info}")

                        if "Không tìm thấy chương trình" in info:
                            ffmpeg_missing = True
                            break
                finally:
                    pbar.update(1)

                    elapsed = time.time() - start_time
                    completed = pbar.n
                    remaining = len(pending_tasks) - completed

                    if completed > 0:
                        avg_per_task = elapsed / completed
                        now = datetime.now()
                        eta = now + timedelta(seconds=avg_per_task * remaining)
                        next_eta = (
                            now + timedelta(seconds=avg_per_task)
                            if remaining > 0 else None
                        )
                        pbar.set_postfix_str(
                            f"ETA={eta.strftime('%H:%M:%S')} | "
                            f"Next={next_eta.strftime('%H:%M:%S') if next_eta else 'Done'}",
                            refresh=True,
                        )
        finally:
            gpu_monitor.stop()

    print("\n========== VIDEO VỪA TẠO THÀNH CÔNG ==========")
    for index, video in enumerate(created_videos, start=1):
        print(
            f"{index}. {video['path']}\n"
            f"   Thời gian render: {video['duration_render']:.1f}s | "
            f"Thời lượng video: {video['duration_video']:.2f}s | "
            f"Packet hình ảnh: {video['packets']}"
        )

    print("\n========== TỔNG KẾT ==========")
    print(f"Tổng task: {len(tasks)}")
    print(f"Video hợp lệ được bỏ qua: {len(skipped_videos)}")
    print(f"Video tạo lại/thành công: {len(created_videos)}")
    print(f"Video thất bại: {len(failed_videos)}")

    if ffmpeg_missing:
        print("Đã dừng vì thiếu FFmpeg hoặc FFprobe.")

    if failed_videos:
        print("\n========== DANH SÁCH LỖI ==========")
        for item in failed_videos:
            print(f"- {item['path']}\n  {item['reason']}")

    print("\nHoàn tất.")


if __name__ == "__main__":
    main()
