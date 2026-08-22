# Hướng dẫn chạy sáu thí nghiệm trên Kaggle cho An, Thanh và Cường

## 1. Mục tiêu

Tài liệu này hướng dẫn ba thành viên chạy toàn bộ sáu tổ hợp thí nghiệm:

| Thành viên | Notebook được giao | GPU 0 | GPU 1 | File kết quả cuối |
|---|---|---|---|---|
| Thanh | `02_run_all_ecapa_tdnn_t4x2.ipynb` | TidyVoice + ECAPA-TDNN | ViMD + ECAPA-TDNN | `who_speak_ai_ecapa_tdnn_resource_constrained.zip` |
| An | `03_run_all_rawnet3_t4x2.ipynb` | TidyVoice + RawNet3 | ViMD + RawNet3 | `who_speak_ai_rawnet3_resource_constrained.zip` |
| Cường | `04_run_all_wavlm_mhfa_t4x2.ipynb` | TidyVoice + WavLM-MHFA | ViMD + WavLM-MHFA | `who_speak_ai_wavlm_mhfa_resource_constrained.zip` |

Thanh được đề xuất chạy ECAPA-TDNN vì môi trường và lỗi deterministic của
ECAPA đã được kiểm tra trên phiên Kaggle của Thanh. Có thể đổi người nếu cần,
nhưng mỗi người chỉ chạy đúng một notebook và không thay đổi tên model trong
notebook.

Ba notebook chạy song song trên ba tài khoản Kaggle riêng. Trong mỗi notebook,
hai bộ dữ liệu tiếp tục chạy song song trên hai GPU T4. Không chia sẻ tài khoản,
mật khẩu, token GitHub, token Hugging Face hoặc Kaggle API key.

## 2. Phạm vi và cách diễn giải kết quả

Đây là cấu hình **resource-constrained** được chốt trước khi có kết quả huấn
luyện thành công, do giới hạn thời gian còn khoảng tám giờ. Đây không phải cấu
hình tối ưu và không thay thế kế hoạch full 15 epoch trong nghiên cứu tương lai.

Cấu hình được sử dụng:

- seed cố định: `42`;
- sử dụng mọi speaker trong Train;
- mỗi epoch lấy một utterance xoay vòng, xác định được, cho mỗi speaker;
- tối đa ba epoch;
- early stopping chỉ dùng Validation, patience bằng một;
- một crop xác định được cho mỗi utterance khi đánh giá;
- batch size đã kiểm tra trên T4: ECAPA `24`, RawNet3 `24`, WavLM-MHFA `6`;
- đánh giá toàn bộ protocol Validation bất biến sau mỗi epoch;
- checkpoint tốt nhất được chọn bằng Validation EER, minDCF là tiêu chí phụ;
- Test chỉ chạy một lần sau huấn luyện bằng checkpoint tốt nhất;
- ngưỡng bảo mật được chọn trên Validation tại FAR `0.1%`, sau đó đóng băng và
  áp dụng nguyên trạng cho Test;
- W&B chạy offline; bằng chứng chính vẫn được lưu bằng JSON, JSONL và checkpoint.

Khi báo cáo, phải gọi đây là **kết quả so sánh có giới hạn tài nguyên, một seed**.
Không được gọi đây là kết quả tối ưu hoặc kết luận tuyệt đối về kiến trúc.

## 3. Revision bắt buộc

Notebook được lưu tại Git commit:

```text
9c011e55b4540b17f8d646d890a3916da1db6b9d
```

Mỗi notebook tự clone và checkout code thực thi đã kiểm thử tại revision:

```text
c68471a69c089cc40a5975b22362da37abcac186
```

Không đổi `PINNED_REVISION`. Không thay bằng `main`, `HEAD` hoặc revision mới
trong lúc ba người đang chạy, vì như vậy sáu thí nghiệm có thể dùng code khác
nhau.

## 4. Điều kiện bắt buộc trước khi chạy

Mỗi thành viên cần đáp ứng đủ các điều kiện sau:

1. Có tài khoản Kaggle riêng và đăng nhập được.
2. Tài khoản còn đủ GPU quota cho một phiên chạy dài.
3. Kaggle cho phép chọn accelerator **GPU T4 x2**.
4. Bật **Internet On** để tải repository và model checkpoint đã pin.
5. Gắn đúng hai Kaggle datasets:
   - `dullahn/mozzila-tidyvoice`;
   - `dullahn/vimd-dataset`.
6. Không chạy notebook model khác trong cùng session.
7. Không thay đổi code, config, seed, split, trial list, batch size, số epoch,
   threshold hoặc đường dẫn output.
8. Không sử dụng notebook pilot cũ hoặc Cell 3 cũ đã bị ngắt.
9. Đảm bảo máy cá nhân có đủ dung lượng để tải ZIP kết quả. ZIP có thể lớn vì
   chứa checkpoint tốt nhất, checkpoint cuối và trạng thái optimizer.
10. Dùng mạng ổn định khi tải file kết quả; không đóng tab trước khi xác nhận
    phiên chạy đã hoàn tất hoặc đã được Kaggle lưu thành version.

Nếu không thấy tùy chọn **GPU T4 x2**, không chọn GPU đơn, P100 hoặc CPU để thay
thế. Dừng lại và báo cho Thanh, vì batch size và phương án song song được kiểm
tra cho hai T4.

## 5. Tải notebook đúng từ GitHub

### 5.1. Cách tải qua giao diện GitHub

1. Mở repository:
   `https://github.com/beaver-felix/Who-Speak-AI`.
2. Chọn branch `thanhDT`.
3. Mở thư mục `model/Thanh2/notebooks`.
4. Mở đúng notebook được phân công.
5. Kiểm tra URL hoặc lịch sử file thuộc commit
   `9c011e55b4540b17f8d646d890a3916da1db6b9d`.
6. Chọn **Download raw file** hoặc **Raw**, rồi lưu file với đuôi `.ipynb`.
7. Không copy từng cell bằng tay.

### 5.2. Liên kết cố định cho từng người

- Thanh — ECAPA-TDNN:
  `https://github.com/beaver-felix/Who-Speak-AI/blob/9c011e55b4540b17f8d646d890a3916da1db6b9d/model/Thanh2/notebooks/02_run_all_ecapa_tdnn_t4x2.ipynb`
- An — RawNet3:
  `https://github.com/beaver-felix/Who-Speak-AI/blob/9c011e55b4540b17f8d646d890a3916da1db6b9d/model/Thanh2/notebooks/03_run_all_rawnet3_t4x2.ipynb`
- Cường — WavLM-MHFA:
  `https://github.com/beaver-felix/Who-Speak-AI/blob/9c011e55b4540b17f8d646d890a3916da1db6b9d/model/Thanh2/notebooks/04_run_all_wavlm_mhfa_t4x2.ipynb`

### 5.3. Kiểm tra SHA-256 sau khi tải

Giá trị đúng:

| Notebook | SHA-256 |
|---|---|
| ECAPA-TDNN | `85c39dba687b96ee0723263843329bf3ae32d53010858c0cc2bbe6eb0445ae11` |
| RawNet3 | `aeac4533abeec15ac450944f24bd72462a39f0946851a7b68decd694d4287529` |
| WavLM-MHFA | `81b6e2561ac27b8cfe98b7444816f62e64f6fd840de5bb2531c492839544f46c` |

Trên PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 ".\ten_notebook.ipynb"
```

Nếu hash khác, xóa file và tải lại từ liên kết cố định. Không chạy file có hash
khác.

## 6. Import notebook vào Kaggle

1. Đăng nhập Kaggle bằng tài khoản cá nhân.
2. Mở khu vực **Code**.
3. Tạo notebook mới hoặc chọn chức năng import/upload notebook.
4. Upload đúng file `.ipynb` đã được phân công.
5. Kiểm tra notebook có đúng ba cell:
   - một cell Markdown mô tả;
   - một cell setup;
   - một cell chạy worker.
6. Không thêm, xóa, sửa hoặc đổi thứ tự cell.
7. Có thể đổi tiêu đề notebook trên Kaggle để dễ nhận biết, ví dụ:
   - `Who Speak AI - Thanh - ECAPA`; hoặc
   - `Who Speak AI - An - RawNet3`; hoặc
   - `Who Speak AI - Cuong - WavLM MHFA`.

Việc đổi tiêu đề Kaggle không thay đổi nội dung thí nghiệm.

## 7. Cấu hình Kaggle session

### 7.1. Accelerator

Trong **Session options** hoặc **Notebook settings**:

1. Chọn accelerator **GPU T4 x2**.
2. Không chọn GPU đơn.
3. Không chọn CPU.
4. Không chọn TPU.

Notebook sẽ tự kiểm tra:

```text
torch.cuda.is_available() == True
torch.cuda.device_count() >= 2
```

Nếu điều kiện không đạt, notebook dừng trước huấn luyện để tránh tạo kết quả
không đồng nhất.

### 7.2. Internet

Bật **Internet On**. Đây là điều kiện bắt buộc để:

- clone GitHub repository;
- checkout revision đã pin;
- cài các dependency còn thiếu;
- tải model checkpoint từ nguồn đã pin.

Cảnh báo Hugging Face về request không có `HF_TOKEN` là chấp nhận được. Không
cần nhập token cá nhân nếu download vẫn tiếp tục.

### 7.3. Persistence

Không phụ thuộc vào persistence để bảo đảm tính đúng. Toàn bộ output cần thiết
được ghi vào `/kaggle/working` và đóng gói thành ZIP cuối cùng. Nếu có tùy chọn
persistence và nhóm chưa thống nhất cách dùng, giữ thiết lập mặc định.

## 8. Gắn hai dataset

1. Trong panel **Input**, chọn **Add Input**.
2. Tìm `dullahn/mozzila-tidyvoice` và thêm dataset.
3. Tiếp tục tìm `dullahn/vimd-dataset` và thêm dataset.
4. Xác nhận panel Input hiển thị cả **Mozilla-TidyVoice** và **ViMDialects**
   hoặc tên tương đương của hai dataset.

Kaggle có thể gom cả hai dataset dưới một publisher mount. Vì vậy
`/kaggle/input` đôi khi chỉ có một thư mục `datasets`; đây không phải lỗi.
Notebook kiểm tra hai đường dẫn thực tế:

```text
/kaggle/input/datasets/dullahn/mozzila-tidyvoice/TidyVoiceX_ASV
/kaggle/input/datasets/dullahn/vimd-dataset
```

Nếu cell setup báo thiếu một trong hai đường dẫn:

1. không sửa đường dẫn trong notebook;
2. mở lại **Add Input**;
3. gỡ dataset sai nếu có;
4. thêm đúng dataset theo slug ở trên;
5. chạy lại từ đầu.

## 9. Cách chạy được khuyến nghị: Save & Run All

Phương án ưu tiên là tạo một Kaggle version chạy sạch từ đầu:

1. Kiểm tra lại GPU T4 x2, Internet On và hai Input.
2. Chọn **Save Version**.
3. Chọn chế độ tương đương **Save & Run All** hoặc **Run All (Commit)**.
4. Đặt ghi chú version, ví dụ:
   - `resource-constrained ECAPA seed 42`;
   - `resource-constrained RawNet3 seed 42`;
   - `resource-constrained WavLM-MHFA seed 42`.
5. Bắt đầu lưu và chạy.
6. Không tạo một phiên chạy thứ hai cho cùng notebook khi phiên đầu còn chạy.
7. Theo dõi trạng thái version/session cho đến khi Kaggle báo hoàn tất.

Ưu điểm:

- chạy lại notebook sạch từ cell đầu;
- output gắn với một version cụ thể;
- giảm rủi ro quên lưu sau khi chạy xong;
- thuận tiện chia sẻ bằng chứng với nhóm.

Nhược điểm:

- log có thể không cập nhật thuận tiện như phiên interactive;
- nếu version thất bại, cần mở log của version để xác định nguyên nhân.

## 10. Cách chạy thay thế: Run All trong phiên interactive

Chỉ dùng khi cần theo dõi log trực tiếp:

1. Chọn **Run All**.
2. Giữ session hoạt động.
3. Không bấm Stop khi đang quét manifest, train, Validation, Test hoặc đóng ZIP.
4. Sau khi thấy thông báo `COMPLETE`, chọn **Save Version** để lưu notebook và
   output.
5. Tải ZIP trước khi chủ động dừng hoặc reset session.

Nếu dùng cách này, việc đóng trình duyệt, reset session hoặc hết quota có thể
làm mất nội dung chưa được Kaggle lưu.

## 11. Diễn tiến mong đợi

### 11.1. Cell setup

Cell đầu tiên có code sẽ:

1. clone repository nếu chưa tồn tại;
2. fetch đúng revision thực thi;
3. checkout detached tại revision đó;
4. xác minh `HEAD` trùng revision đã pin;
5. cài project và đúng dependency của model;
6. kiểm tra có ít nhất hai CUDA GPU;
7. kiểm tra hai dataset mount.

Thông báo đúng ở cuối cell:

```text
PINNED ECAPA WORKER READY c68471a69c089cc40a5975b22362da37abcac186
```

hoặc:

```text
PINNED RAWNET3 WORKER READY c68471a69c089cc40a5975b22362da37abcac186
```

hoặc:

```text
PINNED WAVLM+MHFA WORKER READY c68471a69c089cc40a5975b22362da37abcac186
```

### 11.2. Cell worker

Cell cuối sẽ:

1. tạo sáu config có fingerprint rồi chọn hai config đúng model;
2. với ECAPA, chạy lại strict deterministic gradient gate trước huấn luyện;
3. khởi chạy TidyVoice trên `cuda:0`;
4. khởi chạy ViMD trên `cuda:1`;
5. in trạng thái hai process khoảng mỗi 30 giây;
6. train tối đa ba epoch, Validation sau mỗi epoch;
7. khôi phục checkpoint tốt nhất;
8. chọn threshold trên Validation tại FAR `0.1%`;
9. chạy Test đúng một lần với threshold đã đóng băng;
10. kiểm tra config, metric, trial fingerprint, checkpoint SHA-256 và Test;
11. xóa model cache có thể tải lại;
12. đóng gói toàn bộ bằng chứng cần thiết thành một ZIP.

Trong lúc TidyVoice quét nhiều file, có thể có khoảng thời gian ít log. Không
ngắt chỉ vì chưa thấy batch training ngay lập tức. Scanner đã được tối ưu để
không gọi metadata stat riêng cho 321.711 WAV files.

## 12. Cảnh báo có thể chấp nhận

Các cảnh báo sau không phải lỗi nếu tiến trình vẫn tiếp tục:

- Hugging Face báo request unauthenticated hoặc đề nghị `HF_TOKEN`;
- W&B báo đang chạy `offline`;
- PyTorch báo `weight_norm` deprecated;
- pip báo package đã được cài;
- progress bar tải checkpoint đứng ngắn hạn;
- Kaggle hiển thị hai dataset dưới cùng publisher directory.

Các dấu hiệu sau là lỗi và phải báo ngay:

- `Traceback`;
- `RuntimeError`;
- `CalledProcessError`;
- `FileNotFoundError`;
- CUDA out of memory;
- chỉ có một GPU;
- dataset mount không tồn tại;
- checkpoint hash mismatch;
- trial fingerprint mismatch;
- final evidence validation failed;
- phiên kết thúc nhưng không có ZIP cuối.

## 13. Tiêu chuẩn hoàn tất

Một notebook chỉ được xem là hoàn tất khi log có toàn bộ điều kiện sau:

1. Cell setup in đúng revision đã pin.
2. Cả TidyVoice và ViMD process đều kết thúc với mã `0`.
3. Có hai thông báo `TRAINING RUN EVIDENCE VALIDATED`.
4. Có thông báo:

```text
RESOURCE-CONSTRAINED WORKER COMPLETE
```

5. Có thông báo `DOWNLOAD READY` trỏ đến đúng ZIP.
6. Cell cuối in:

```text
COMPLETE - download from Kaggle Output panel: ...
```

7. Kaggle version có trạng thái hoàn tất, không phải failed hoặc cancelled.

Không nghiệm thu dựa riêng vào việc cell có màu xanh; phải kiểm tra thông báo
và ZIP.

## 14. Tải và lưu toàn bộ kết quả

Sau khi phiên chạy thành công:

1. Mở panel **Output** của notebook hoặc version.
2. Tìm đúng file ZIP theo bảng phân công.
3. Chọn tải file về máy.
4. Không đổi nội dung ZIP.
5. Có thể đổi tên bản sao ngoài ZIP để thêm tên người chạy, nhưng phải giữ một
   bản với tên gốc.
6. Tính SHA-256 cho ZIP sau khi tải:

```powershell
Get-FileHash -Algorithm SHA256 ".\who_speak_ai_<model>_resource_constrained.zip"
```

7. Gửi cho Thanh ba thông tin:
   - tên file;
   - dung lượng file;
   - SHA-256;
   - liên kết Kaggle version đã lưu.
8. Giữ một bản sao dự phòng trên ổ đĩa hoặc Drive của nhóm cho đến khi dự án
   được nộp.

Thư mục tập hợp đề xuất trên máy Thanh:

```text
Who-Speak-AI/model/Thanh2/results/resource_constrained/downloads/
```

Các ZIP và checkpoint lớn không được commit vào Git. Sau khi nhận đủ ba ZIP,
Thanh sẽ chạy validator, trích metric JSON cần thiết, tạo bảng so sánh và chỉ
commit các artifact nhỏ phù hợp.

## 15. Nội dung tối thiểu bên trong ZIP

ZIP hợp lệ phải có thư mục của model và tối thiểu:

```text
<model>/
├── configs/
├── logs/
│   ├── tidyvoice.log
│   └── vimd.log
├── runs/
│   ├── tidyvoice/
│   │   ├── checkpoints/
│   │   │   ├── best.pt
│   │   │   ├── best.pt.json
│   │   │   ├── last.pt
│   │   │   └── last.pt.json
│   │   ├── validation/
│   │   ├── final_test.json
│   │   ├── metrics.jsonl
│   │   ├── resolved_config.json
│   │   └── run_summary.json
│   └── vimd/
│       └── ... cùng cấu trúc ...
└── worker_manifest.json
```

ECAPA còn có `ecapa_strict_determinism_gate.json`.

Không được xóa `best.pt`, `last.pt`, sidecar JSON, `final_test.json`,
`run_summary.json` hoặc log trước khi gửi kết quả.

## 16. Xử lý khi bị gián đoạn

### 16.1. Chỉ bấm Stop nhưng session Kaggle vẫn còn

Nếu `/kaggle/working` vẫn tồn tại và đã có
`runs/<dataset>/checkpoints/last.pt`, chạy lại **cell worker cuối**. Script sẽ tự
thêm `--resume` cho run có checkpoint.

Không xóa run directory và không chạy lại cell setup nếu repository vẫn sạch.

### 16.2. Session đã reset hoặc bị đóng hoàn toàn

Khi `/kaggle/working` đã mất, checkpoint chưa được lưu vào một Kaggle version
cũng mất. Khi đó:

1. tạo session mới;
2. cấu hình lại GPU T4 x2, Internet và hai datasets;
3. chạy lại toàn bộ notebook từ đầu.

Không được khai báo là resume nếu checkpoint cũ không còn.

### 16.3. Một dataset thất bại, dataset còn lại thành công

Worker để process còn lại hoàn thành nhằm tránh mất kết quả. Tuy nhiên ZIP cuối
không được tạo nếu một process thất bại. Thực hiện:

1. lưu Kaggle version hiện tại;
2. mở `logs/tidyvoice.log` và `logs/vimd.log` trong Output;
3. gửi log lỗi cuối cùng cho Thanh;
4. không tự sửa code hoặc config;
5. chỉ chạy lại sau khi nhóm xác định nguyên nhân.

### 16.4. Repository báo dirty

Nếu thấy:

```text
Existing worker repository is dirty; start a fresh Kaggle session.
```

Không dùng `git reset --hard` và không xóa tùy tiện. Mở session Kaggle mới rồi
chạy lại để tránh làm mất artifact chưa tải.

## 17. Thông tin mỗi thành viên phải báo cho nhóm

Khi bắt đầu:

```text
Tên:
Model:
Kaggle notebook/version URL:
GPU: T4 x2
Hai datasets đã gắn: Có/Không
Thời gian bắt đầu:
```

Khi hoàn tất:

```text
Tên:
Model:
Trạng thái TidyVoice: Thành công/Thất bại
Trạng thái ViMD: Thành công/Thất bại
Tên ZIP:
Dung lượng ZIP:
SHA-256 ZIP:
Kaggle version URL:
Thời gian hoàn tất:
```

Nếu thất bại, bổ sung:

```text
Dataset/process bị lỗi:
Dòng lỗi cuối:
Đã lưu Kaggle version: Có/Không
Đường dẫn hoặc file log:
```

## 18. Checklist nhanh

### Trước khi chạy

- [ ] Đúng notebook được phân công.
- [ ] Notebook hash đúng.
- [ ] GPU T4 x2.
- [ ] Internet On.
- [ ] Đã gắn TidyVoice.
- [ ] Đã gắn ViMD.
- [ ] Không sửa cell hoặc revision.
- [ ] Không có notebook model khác đang chạy trong cùng session.
- [ ] Có đủ quota và dung lượng tải ZIP.

### Sau khi chạy

- [ ] Cả hai process thành công.
- [ ] Hai run evidence validators thành công.
- [ ] Có `RESOURCE-CONSTRAINED WORKER COMPLETE`.
- [ ] Có ZIP đúng tên.
- [ ] Đã lưu Kaggle version.
- [ ] Đã tải ZIP.
- [ ] Đã tính SHA-256 ZIP.
- [ ] Đã gửi ZIP, hash và version URL cho Thanh.
- [ ] Không commit ZIP/checkpoint lớn lên Git.

## 19. Ưu điểm và hạn chế của phương án

### Ưu điểm

- Sáu tổ hợp được chạy đồng thời trên sáu GPU T4.
- Teammate chỉ cần import notebook, cấu hình Kaggle và chọn Run All.
- Mỗi notebook pin cùng một code revision và tự kiểm tra môi trường.
- TidyVoice và ViMD được cô lập theo process, GPU, cache, log và run directory.
- Validation, final Test, checkpoint và threshold provenance được kiểm tra tự
  động trước khi tạo ZIP.
- Có thể resume nếu checkpoint vẫn còn trong cùng session.

### Hạn chế

- Ba epoch có thể chưa đủ hội tụ.
- Một seed không đo được độ biến thiên giữa các lần chạy.
- Patience một có thể nhạy với dao động Validation.
- Ba tài khoản Kaggle có thể có khác biệt nhỏ về thời gian và hạ tầng.
- Nếu session mất trước khi checkpoint được lưu thành version, phải chạy lại.
- ZIP checkpoint, đặc biệt WavLM-MHFA, có thể lớn và mất thời gian tải.
