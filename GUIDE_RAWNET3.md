# Hướng dẫn bàn giao và tích hợp RawNet3 vào ứng dụng
1. Artifact phải dùng
Model được chọn là RawNet3 fine-tune trên ViMD, best Validation epoch 2 (epoch thứ ba vì artifact đánh index từ 0).

Lấy đúng file sau từ ZIP:

ZIP: https://drive.google.com/file/d/1GvvGbe1bOwXTG2fj7BziDuyQfBYlZ409/view?usp=sharing

model/Thanh2/results/team_runs/who_speak_ai_rawnet3_resource_constrained.zip

Checkpoint bên trong ZIP:

rawnet3/runs/vimd/checkpoints/best.pt

Checkpoint sidecar:

rawnet3/runs/vimd/checkpoints/best.pt.json

Resolved config:

rawnet3/runs/vimd/resolved_config.json

Final evidence:

rawnet3/runs/vimd/final_test.json

SHA-256 bắt buộc:

ZIP:

593f416cbf68025f09d6415b6d00bdc9f444382e3c1d9396f1426951b9658253

best.pt:

0b06fd3c4644d6c1cf4e7f9c087cf7fba8be493952589ab4321f8319d4215386

Không dùng:

last.pt thay cho best.pt;
RawNet3 pretrained model.pt thay cho checkpoint đã fine-tune;
checkpoint TidyVoice;
AAM classifier head để xác thực user trong app;
threshold ví dụ 0.75 trong tài liệu concept cũ.
2. Giải nén và kiểm tra trên PowerShell
Chạy từ repository root:

$zip = ".\model\Thanh2\results\team_runs\who_speak_ai_rawnet3_resource_constrained.zip"

$destination = ".\model\Thanh2\results\selected_model"

Get-FileHash -Algorithm SHA256 $zip

Expand-Archive -LiteralPath $zip -DestinationPath $destination

$checkpoint = Join-Path $destination "rawnet3\runs\vimd\checkpoints\best.pt"

Get-FileHash -Algorithm SHA256 $checkpoint

Chỉ tiếp tục khi hai hash khớp mục 1. Giữ best.pt.json, resolved_config.json và final_test.json cùng artifact bàn giao để truy vết nguồn gốc.
3. Source và môi trường
Run RawNet3 được tạo bằng code tại revision:

c68471a69c089cc40a5975b22362da37abcac186

Thông tin runtime đã xác thực:

Python 3.12 trên Kaggle;
PyTorch 2.10.0+cu128;
CUDA build 12.8;
asteroid-filterbanks==0.4.0;
huggingface-hub>=1.11,<1.12;
model checkpoint source jungjee/RawNet3, revision c89102eea20c3f96917c434de673c0ace0caddc0;
architecture source clovaai/voxceleb_trainer, revision f51bab870672a9b0b50fa158b4e30f329e7866d7.

PyTorch không nằm trong dependency group vì phải cài build phù hợp CPU/CUDA của máy triển khai trước. Sau đó, tại model/Thanh2:

python -m pip install -e ".[data,rawnet3]"

Nếu dùng CPU, model vẫn có thể chạy nhưng dự án chưa benchmark latency CPU và threshold hiện tại được đo bằng CUDA FP16. Phải hiệu chuẩn lại threshold bằng dữ liệu app trước khi gọi là production-ready.
4. Contract inference bắt buộc
Thành phần
Giá trị
Sample rate
16.000 Hz
Channels
Mono; nhiều kênh lấy trung bình số học
Resampling
SciPy polyphase
Amplitude normalization
Không dùng
Evaluation segment
64.240 mẫu = 4,015 giây
Segment count hiện tại
1
Short audio
Lặp waveform đến đủ độ dài
Embedding
256 chiều, L2-normalized
Score
Cosine similarity
Accept rule
score >= threshold
SV threshold ban đầu
0.6565317440180312


Không thay đổi preprocessing nhưng giữ nguyên threshold, vì như vậy score distribution không còn tương ứng với Validation đã hiệu chuẩn.
5. Loader inference tham khảo
Đoạn mã sau dùng đúng adapter, preprocessing, crop và checkpoint structure của training pipeline. Đội app nên đưa class này vào service/model layer, không đặt trực tiếp trong UI callback.

"""Load the selected RawNet3 checkpoint and produce speaker embeddings."""

from __future__ import annotations

import hashlib

from collections.abc import Mapping, Sequence

from pathlib import Path

import numpy as np

import soundfile as sf

import torch

import torch.nn.functional as functional

from speaker_recognition.data.audio import canonicalize_audio

from speaker_recognition.data.segments import evenly_spaced_segments

from speaker_recognition.models.rawnet3 import (

    RAWNET3_EVALUATION_SAMPLES,

    RawNet3Adapter,

)


SELECTED_CHECKPOINT_SHA256 = (

    "0b06fd3c4644d6c1cf4e7f9c087cf7fba8be493952589ab4321f8319d4215386"

)

SELECTED_CONFIG_SHA256 = (

    "08392e8146fb4ae42cc7b971641d4339c9f98f517bd6f3ece1d5b2bf2fdafa94"

)

SV_THRESHOLD = 0.6565317440180312


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:

    """Hash a model file incrementally so a corrupted artifact is rejected."""

    digest = hashlib.sha256()

    with path.open("rb") as stream:

        while chunk := stream.read(chunk_bytes):

            digest.update(chunk)

    return digest.hexdigest()


class RawNet3SpeakerEncoder:

    """Wrap the selected fine-tuned encoder behind one inference contract."""

    def __init__(

        self,

        checkpoint_path: str | Path,

        *,

        cache_dir: str | Path,

        device: str = "cuda:0",

    ) -> None:

        """Authenticate, safely load, and freeze the selected RawNet3 model."""

        self.device = torch.device(device)

        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()

        if sha256_file(self.checkpoint_path) != SELECTED_CHECKPOINT_SHA256:

            raise RuntimeError("Selected RawNet3 checkpoint SHA-256 mismatch.")

        # This constructs the exact pinned architecture. The source checkpoint

        # is authenticated by RawNet3Adapter before fine-tuned weights replace it.

        self.adapter = RawNet3Adapter.from_pretrained(

            cache_dir=cache_dir,

            device=str(self.device),

        )

        payload = torch.load(

            self.checkpoint_path,

            map_location="cpu",

            weights_only=True,

        )

        if not isinstance(payload, Mapping):

            raise RuntimeError("Training checkpoint root must be a mapping.")

        identity = payload.get("identity")

        expected_identity = {

            "model_name": "rawnet3",

            "dataset_name": "vimd",

            "config_sha256": SELECTED_CONFIG_SHA256,

            "manifest_sha256": (

                "ed7b764c6aaab2ba2c4ec95edadab19fd640ebca72aa06da3d36cbf93fc4747f"

            ),

            "seed": 42,

        }

        if identity != expected_identity:

            raise RuntimeError("Checkpoint experiment identity mismatch.")

        adapter_state = payload.get("adapter_state")

        if not isinstance(adapter_state, Mapping):

            raise RuntimeError("Checkpoint has no adapter state mapping.")

        self.adapter.load_state_dict(adapter_state, strict=True)

        self.adapter.to(self.device)

        self.adapter.eval()

        self.adapter.requires_grad_(False)

    def embed_file(self, audio_path: str | Path) -> np.ndarray:

        """Return one normalized 256-D embedding from an audio file."""

        waveform, sample_rate = sf.read(

            Path(audio_path),

            dtype="float32",

            always_2d=True,

        )

        canonical = canonicalize_audio(

            waveform,

            sample_rate=int(sample_rate),

            target_sample_rate=16000,

        )

        segments = evenly_spaced_segments(

            canonical,

            num_samples=RAWNET3_EVALUATION_SAMPLES,

            segment_count=1,

        )

        batch = torch.from_numpy(segments).to(self.device)

        with torch.inference_mode():

            if self.device.type == "cuda":

                # This matches the accepted evaluation precision.

                with torch.autocast(

                    device_type="cuda",

                    dtype=torch.float16,

                    enabled=True,

                ):

                    crop_embeddings = self.adapter(batch)

            else:

                crop_embeddings = self.adapter(batch)

            utterance_embedding = functional.normalize(

                crop_embeddings.float().mean(dim=0, keepdim=True),

                p=2,

                dim=1,

            )

        return utterance_embedding.squeeze(0).cpu().numpy()


def average_enrollment(embeddings: Sequence[np.ndarray]) -> np.ndarray:

    """Average multiple recordings and L2-normalize one user template."""

    if len(embeddings) < 3:

        raise ValueError("Enrollment requires at least three recordings.")

    matrix = np.stack(embeddings).astype(np.float32, copy=False)

    template = matrix.mean(axis=0)

    norm = float(np.linalg.norm(template))

    if not np.isfinite(norm) or norm <= 0.0:

        raise ValueError("Enrollment template has invalid norm.")

    return np.ascontiguousarray(template / norm, dtype=np.float32)


def cosine_score(query: np.ndarray, template: np.ndarray) -> float:

    """Score two normalized embeddings using their dot product."""

    score = float(np.dot(query, template))

    if not np.isfinite(score):

        raise ValueError("Cosine score is not finite.")

    return score
6. Enrollment
Quy trình tối thiểu cho một user:

Thu ba recording riêng, mỗi recording nên chứa speech rõ và dài ít nhất khoảng 4,015 giây để tránh phải lặp audio.
Kiểm tra file decode được, không rỗng và không có sample không hữu hạn.
Chạy embed_file() cho từng recording.
Lấy trung bình ba embedding rồi L2-normalize bằng average_enrollment().
Lưu template 256 chiều cùng metadata:
user_id;
tên hiển thị;
model rawnet3-vimd-best-epoch-2;
checkpoint SHA-256;
preprocessing version;
thời điểm enrollment.
Không lưu raw audio nếu use case không cần; nếu lưu phải có đồng ý của user, phân quyền truy cập và chính sách xóa.

JSON có thể lưu embedding dạng list cho demo. Với hệ thống lớn hơn, dùng SQLite và BLOB/array storage. Không dùng AAM class index làm user ID: head training chỉ phục vụ học embedding và không đại diện cho user mới của app.
7. Speaker Verification 1:1
query = encoder.embed_file("query.wav")

score = cosine_score(query, enrolled_template)

verified = score >= SV_THRESHOLD

Ngưỡng 0.6565317440180312 được chọn trên ViMD Validation tại FAR mục tiêu 0,1%, sau đó đóng băng. Trên ViMD Test, ngưỡng này đạt:

FAR 0,075%;
FRR 15,547%;
TAR 84,453%.

Đây là ngưỡng khởi tạo có evidence, không phải đảm bảo production. Microphone, nhiễu, codec, passphrase và nhóm user của app có thể làm score distribution thay đổi. Khi có dữ liệu app, chọn lại threshold trên một app-validation set; không chỉnh bằng app-test hoặc vài case demo mong muốn.
8. Speaker Identification 1:N
query embedding

  -> cosine với mọi enrollment template

  -> chọn score lớn nhất

  -> nếu score >= SID unknown threshold: trả user tương ứng

  -> ngược lại: guest/unknown

Threshold hiện tại được hiệu chuẩn cho SV 1:1, chưa được xác thực cho open-set SID 1:N. Có thể dùng nó làm giá trị demo ban đầu nhưng phải ghi là provisional. Threshold SID cần được chọn riêng trên dữ liệu validation gồm known và unknown speakers của app; khi số user tăng, xác suất false identification cũng thay đổi.
9. Gắn vào luồng ứng dụng
Audio input

  -> canonical preprocessing

  -> RawNet3 embedding

  -> SID 1:N

       -> known: lấy profile, cá nhân hóa prompt

       -> unknown: guest mode

  -> intent/router

       -> tác vụ thường: thực thi

       -> nhật ký cá nhân: SV 1:1 với claimed/identified user

            -> pass: thực thi

            -> fail: từ chối

Model nên được load đúng một lần khi service khởi động. Không load checkpoint lại sau mỗi request. Enrollment templates có thể cache trong RAM nhưng database vẫn là nguồn dữ liệu chính.
10. Acceptance checklist cho đội app
Hash ZIP và best.pt khớp.
Dùng RawNet3/ViMD best.pt, không dùng last.pt hoặc base model.pt.
Restricted load với weights_only=True và strict=True.
Model ở eval() và inference dùng torch.inference_mode().
Preprocessing đúng mono float32 16 kHz, không amplitude normalization.
Embedding cuối có shape (256,), hữu hạn và norm xấp xỉ 1.
Enrollment trung bình tối thiểu ba recording rồi normalize lại.
Cosine score hữu hạn; accept rule là >=.
SV threshold ban đầu đúng 0.6565317440180312.
SID có nhánh unknown/guest; không ép mọi giọng nói vào một user.
Model chỉ load một lần.
Có test same-speaker, different-speaker, unknown, audio ngắn, stereo và sample rate khác 16 kHz.
Có đo end-to-end latency trên máy thật; không dùng latency T4 làm latency app.
Không dùng dữ liệu Test để điều chỉnh threshold.
11. Ưu và nhược điểm của lựa chọn
Ưu điểm:

Tốt nhất trên ViMD trong bốn run được chấp nhận.
16,28 triệu tham số, ít hơn ECAPA 21,6%.
Batch-1 model-only median 8,840 ms trên T4, nhanh hơn ECAPA trong cùng gate.
Một embedding dùng được cho cả SV và SID.
Checkpoint, config và threshold có hash/provenance rõ ràng.

Nhược điểm:

Checkpoint training khoảng 227 MB vì chứa optimizer và head, lớn hơn encoder inference tối thiểu.
Chưa benchmark CPU, mobile hoặc end-to-end app.
Threshold chưa hiệu chuẩn trên microphone và user population của app.
RawNet3 kém ECAPA trên TidyVoice, nên lựa chọn phụ thuộc target tiếng Việt.
