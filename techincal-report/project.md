# **Final Project** 

### **Secure Virtual Assistant with Speaker Recognition** 

## **1 Overview** 

   - In recent years, virtual assistants have become increasingly popular and have been adopted across a wide range of domains. These systems can support users in performing various tasks, such as retrieving information, setting reminders, controlling devices, accessing personal data, and modifying system settings. 

   - However, not every function of a virtual assistant should be executed immediately without additional verification. For sensitive or important tasks, the system must authenticate the user before allowing the requested action to be performed. In addition, speaker identification can be used to personalize the user experience for appropriate functions. 

   - In this project, students will develop a virtual assistant that integrates speaker verification for important tasks and speaker identification for personalization. Specifically, students will select a speaker verification or speaker identification model, train and evaluate it using a suitable dataset, and integrate the model into a complete virtual assistant application. 

## **2 Requirements** 

1. **(5 points)** Students must select a representative model for speaker verification or speaker identification, such as ECAPA-TDNN, RawNet3, or an equivalent model, and use a suitable dataset to train and evaluate it. The report must clearly describe the dataset, the train, validation, and test data splits, the selected model, the training procedure, appropriate evaluation metrics, and the experimental results. 

2. **(5 points)** Using the model developed in Requirement 1, students must build a complete virtual assistant that integrates speaker verification and speaker identification. The system must be designed around specific use cases proposed by the students and satisfy the following requirements: 

   - The virtual assistant must support voice-based interaction with users. The system must also provide a speaker enrollment and management component, such as a web interface or a simple application, for collecting initial voice data and managing the information required for speaker verification and identification. 

   - The system must include at least one general function that does not require authentication, at least one important function that may only be performed after successful speaker verification, and at least one function that uses speaker identification to personalize responses, information, or settings for each registered user. 

   - For the speaker verification and identification features, students must clearly describe the enrollment procedure for a new user, including how the initial voice data is collected and how the information required for speaker verification and identification is stored. 

   - The report must clearly present the overall system architecture and processing flow. 

## **3 Suggested Virtual Assistant Implementation** 

   - A typical virtual assistant architecture may be organized as a pipeline consisting of speech signal acquisition, Automatic Speech Recognition (ASR), request analysis and task orchestration, and Text-to-Speech (TTS) for generating spoken responses. For important tasks, the system must include a speaker verification (SV) step before executing the corresponding action. In addition, a speaker identification (SID) component may be incorporated to support personalization for appropriate functions. 

   - The request analysis and task orchestration module may be implemented using one of the following approaches: 

- **Rule-based:** Develop a set of rules and command patterns that directly map user inputs to corresponding tasks. 

- **Intent–Entity-based:** Formulate the problem as intent classification and entity extraction, followed by mapping the identified intent and entities to the corresponding action. Frameworks such as Rasa may be used to implement this approach. 

- **LLM-based:** Use Large Language Models (LLMs) to infer user intent and orchestrate the corresponding tasks. 

## **4 Submission Guidelines** 

- Students must submit the complete source code, trained model weights, training dataset, and project report. 

- All submission materials must be packaged into a single ZIP file. The filename must contain the student IDs of all team members, separated by underscores, using the following format: 

```
StudentID1StudentID2...StudentIDN.zip
```


- If the dataset or trained model is too large to include in the ZIP file, students may upload the materials to Google Drive and submit a text file containing the corresponding link. The text file must be named using the following format: 

```
StudentID1StudentID2...StudentIDN.txt
```

# train 3 model, đánh giá

# Báo cáo huấn luyện và đánh giá ECAPA-TDNN, RawNet3

## 1\. Kết luận ngắn

- Đã hoàn thành và xác thực bốn thí nghiệm: hai mô hình × hai bộ dữ liệu.  
- ECAPA-TDNN tốt hơn trên TidyVoice.  
- RawNet3 tốt hơn trên ViMD, có ít tham số hơn và nhanh hơn trong phép đo model-only batch 1 trên Tesla T4.  
- Chọn **RawNet3 fine-tune trên ViMD** để bàn giao cho ứng dụng tiếng Việt.  
- Đây là kết quả **resource-constrained, một seed**, không phải kết quả tối ưu kiến trúc hoặc tối ưu siêu tham số.

Checkpoint được chọn:

Archive: results/team\_runs/who\_speak\_ai\_rawnet3\_resource\_constrained.zip

Member:  rawnet3/runs/vimd/checkpoints/best.pt

SHA-256: 0b06fd3c4644d6c1cf4e7f9c087cf7fba8be493952589ab4321f8319d4215386

Best Validation epoch: 2 (epoch thứ ba do index bắt đầu từ 0\)

## 2\. Mục tiêu và phạm vi

Mục tiêu là xây dựng speaker embedding phục vụ:

- Speaker Verification (SV): so sánh 1:1 giữa giọng nói truy vấn và mẫu đã đăng ký.  
- Speaker Identification (SID): so sánh 1:N với các speaker đã đăng ký.

Ban đầu nhóm khảo sát ECAPA-TDNN, RawNet3 và WavLM+MHFA. WavLM+MHFA bị loại khỏi kết quả cuối vì các lần chạy không tạo được evidence hoàn chỉnh: lần đầu gặp embedding FP16 không hữu hạn trong training; lần sau ViMD Final Test bị từ chối vì evidence chứa số không hữu hạn. Không sử dụng checkpoint hoặc metric WavLM trong bảng so sánh.

## 3\. Dữ liệu và cách chia tập

### 3.1. TidyVoice

| Split chuẩn hóa | Utterance | Speaker |
| :---- | ----: | ----: |
| Train | 262.268 | 3.666 |
| Validation | 29.720 | 404 |
| Test | 29.723 | 404 |

- Source Train được giữ làm Train.  
- Source Dev có 808 speaker và được chia 404/404 speaker cho Validation/Test.  
- Split dùng seed 42 và tối ưu cân bằng số utterance cùng phân bố ngôn ngữ.  
- Speaker giữa Train, Validation và Test không giao nhau.  
- Validation/Test gần cân bằng tuyệt đối về số utterance: 29.720/29.723.

### 3.2. ViMD

| Split chuẩn hóa | Utterance | Speaker | Genuine-pair capacity |
| :---- | ----: | ----: | ----: |
| Train | 15.023 | 10.291 | 7.044 |
| Validation | 1.898 | 1.318 | 879 |
| Test | 2.026 | 1.344 | 1.046 |

- Source Test được giữ nguyên.  
- Hai speaker trùng giữa source Validation và Test là `spk_73_0186` và `spk_76_0219`; hai hàng singleton tương ứng bị loại khỏi Validation.  
- Việc loại hai hàng này xóa speaker leakage nhưng không làm giảm số genuine pair khả dụng của Validation.  
- Gender chỉ dùng làm metadata mô tả, không làm target vì audit phát hiện nhãn không nhất quán.

### 3.3. Tiền xử lý dùng chung

WAV hoặc audio bytes trong Parquet

  \-\> decode float32

  \-\> downmix nhiều kênh bằng trung bình số học

  \-\> resample polyphase về 16 kHz

  \-\> không amplitude normalization

  \-\> crop xác định được

  \-\> model embedding

- TidyVoice: WAV mono 16 kHz.  
- ViMD: WAV bytes mono/stereo 44,1 hoặc 48 kHz trong Parquet, đọc lazy theo row group.  
- Audio ngắn hơn crop được lặp mẫu, không chèn silence.  
- Crop training được xác định bởi seed, epoch và utterance ID.  
- Evaluation dùng crop cách đều, xác định được.

## 4\. Hai mô hình

| Thuộc tính | ECAPA-TDNN | RawNet3 |
| :---- | :---- | :---- |
| Nguồn pretrained | `speechbrain/spkrec-ecapa-voxceleb` | `jungjee/RawNet3` |
| Revision checkpoint | `0f99f2d...7286` | `c89102e...addc0` |
| Nguồn kiến trúc | SpeechBrain 1.1.0 | Clova VoxCeleb trainer `f51bab8...86d7` |
| Input | waveform 16 kHz → acoustic features | waveform 16 kHz trực tiếp |
| Embedding | 192 chiều | 256 chiều |
| Encoder parameters | 20.767.552 | 16.280.322 |
| Pooling/đặc điểm chính | ECAPA embedding model | Sinc filterbank, residual encoder, ECA pooling |

Các source classifier pretrained bị bỏ. Mỗi dataset có AAM-Softmax head mới với số class đúng bằng số speaker Train. Khi triển khai SV/SID, chỉ dùng encoder embedding; không dùng AAM classifier head.

## 5\. Thiết kế huấn luyện

### 5.1. Phần dùng chung

| Thành phần | Giá trị |
| :---- | :---- |
| Seed | 42 |
| Objective | AAM-Softmax |
| Margin / scale | 0,2 / 30 |
| Precision | FP16, initial loss scale 1024 |
| Gradient clipping | Global norm 5,0 |
| Schedule | Constant learning rate |
| Epoch budget | Tối đa 3 |
| Early stopping | Validation EER, patience 1 |
| Tie-break | Validation minDCF |
| Minimum EER improvement | 0,001 |
| Checkpoint interval | 100 optimizer steps |
| DataLoader workers | 2, `persistent_workers=false` |
| W\&B | Offline; evidence chính là JSON/JSONL/checkpoint |

Mỗi epoch dùng tất cả speaker Train nhưng chỉ lấy một utterance xác định được cho mỗi speaker. Vì vậy:

- TidyVoice: 3.666 training examples/epoch.  
- ViMD: 10.291 training examples/epoch.

Đây là phương án được chốt trước khi có kết quả để đáp ứng giới hạn khoảng tám giờ. Kế hoạch đầy đủ ban đầu là tối đa 15 epoch, tối đa bốn utterance/speaker và patience 3; kế hoạch này không được chạy và không được mô tả như kết quả hiện tại.

### 5.2. Phần theo kiến trúc

| Cấu hình | ECAPA-TDNN | RawNet3 | Lý do |
| :---- | ----: | ----: | :---- |
| Batch size | 24 | 24 | Cả hai pass đến 32 trên T4; chọn 24 để giữ headroom |
| Train crop | 48.000 mẫu (3,00 s) | 48.240 mẫu (3,015 s) | Theo đường xử lý/source recipe từng model |
| Evaluation crop | 48.000 mẫu | 64.240 mẫu (4,015 s) | Cấu hình đã qua integration gate |
| Optimizer | Adam | Adam | Phù hợp recipe pretrained tương ứng |
| Encoder LR | `1e-4` | `1e-4` | Fine-tune bảo thủ từ pretrained |
| Head LR | `1e-4` | `1e-4` | Cùng control ban đầu |
| Weight decay | `2e-6` | `5e-5` | Theo recipe/source model tương ứng |
| Trainable scope | Toàn encoder \+ head mới | Toàn encoder \+ head mới | Target-domain fine-tuning |

Batch 24 không được chọn tùy ý. Memory gate với head ViMD 10.291 class đã pass các batch 4, 8, 16, 24 và 32\. Chính sách chọn không vượt 80% mức lớn nhất dẫn đến candidate 24; sau đó mỗi model pass thêm gate ba batch thật, 72 speaker khác nhau và ba optimizer updates hữu hạn.

Các learning rate là giả thuyết có căn cứ từ source/transfer recipe, không được gọi là tối ưu vì không có grid search hoặc ablation do giới hạn thời gian.

## 6\. Protocol đánh giá

- Mọi model dùng đúng cùng trial list cho từng dataset/split.  
- Genuine pairs được giới hạn tối đa 20/speaker để speaker nhiều dữ liệu không chi phối kết quả.  
- Mỗi split có 100.000 impostor trials, đủ độ phân giải cho FAR 0,01%.  
- Score là cosine similarity giữa hai embedding đã L2-normalize.  
- EER thấp hơn tốt hơn.  
- minDCF thấp hơn tốt hơn, với `P_target=0,01`, `C_miss=C_false_alarm=1`.  
- Báo cáo thêm FAR, FRR, TAR, accuracy và TAR@FAR 5%, 1%, 0,1%, 0,01%.  
- Checkpoint được chọn chỉ bằng Validation EER; minDCF là tie-break.  
- Ngưỡng bảo mật được chọn tại FAR 0,1% trên Validation, đóng băng rồi áp dụng đúng một lần trên Test. Test không được dùng để chọn model trong training hay chỉnh threshold.

Trial counts:

| Dataset | Split | Genuine | Impostor | Tổng |
| :---- | :---- | ----: | ----: | ----: |
| TidyVoice | Validation | 7.954 | 100.000 | 107.954 |
| TidyVoice | Test | 7.898 | 100.000 | 107.898 |
| ViMD | Validation | 863 | 100.000 | 100.863 |
| ViMD | Test | 1.042 | 100.000 | 101.042 |

## 7\. Diễn biến training

| Model / Dataset | Epoch | Train loss | Validation EER | Validation minDCF | Kết quả |
| :---- | ----: | ----: | ----: | ----: | :---- |
| ECAPA / TidyVoice | 0 | 16,4067 | 3,943% | 0,3685 | Best |
| ECAPA / TidyVoice | 1 | 15,7529 | 4,075% | 0,4068 | Dừng sớm |
| ECAPA / ViMD | 0 | 17,0806 | 3,936% | 0,4503 | Best |
| ECAPA / ViMD | 1 | 15,3398 | 6,634% | 0,7661 | Dừng sớm |
| RawNet3 / TidyVoice | 0 | 15,7133 | 5,860% | 0,5251 | Best |
| RawNet3 / TidyVoice | 1 | 14,7825 | 9,693% | 0,7767 | Dừng sớm |
| RawNet3 / ViMD | 0 | 15,9177 | 29,664% | 0,9164 | Cải thiện |
| RawNet3 / ViMD | 1 | 14,8658 | 7,532% | 0,6805 | Cải thiện |
| RawNet3 / ViMD | 2 | 13,3438 | 2,692% | 0,2446 | Best |

Chỉ số epoch trong artifact bắt đầu từ 0: epoch 0 là epoch đầu tiên và epoch 2 là epoch thứ ba.

Training loss giảm không đồng nghĩa Validation luôn tốt hơn. Ba run đầu cho thấy loss giảm nhưng EER xấu đi ở epoch kế tiếp; early stopping đã giữ epoch 0\. RawNet3/ViMD cải thiện liên tục và chọn epoch 2\.

## 8\. Kết quả Final Test

### 8.1. Chỉ số chính

| Model | Dataset | EER ↓ | minDCF ↓ | Accuracy | FAR tại ngưỡng đóng băng | FRR | TAR |
| :---- | :---- | ----: | ----: | ----: | ----: | ----: | ----: |
| ECAPA | TidyVoice | **3,912%** | **0,3586** | 97,865% | 0,080% | 28,159% | **71,841%** |
| RawNet3 | TidyVoice | 5,875% | 0,5624 | 96,509% | 0,124% | 46,126% | 53,874% |
| ECAPA | ViMD | 3,804% | 0,4516 | 99,527% | 0,116% | 34,741% | 65,259% |
| RawNet3 | ViMD | **3,071%** | **0,2262** | **99,765%** | **0,075%** | **15,547%** | **84,453%** |

Accuracy cao vì protocol có 100.000 impostor trials nhưng số genuine nhỏ hơn nhiều; do đó EER, minDCF và TAR tại FAR thấp là chỉ số chính.

### 8.2. TAR tại các FAR mục tiêu

| Model / Dataset | TAR@FAR 5% | TAR@FAR 1% | TAR@FAR 0,1% | TAR@FAR 0,01% |
| :---- | ----: | ----: | ----: | ----: |
| ECAPA / TidyVoice | 96,771% | 90,377% | 73,563% | 48,974% |
| RawNet3 / TidyVoice | 93,391% | 82,907% | 48,088% | 0,886% |
| ECAPA / ViMD | 97,505% | 88,484% | 63,820% | 37,812% |
| RawNet3 / ViMD | 97,217% | **95,106%** | **86,756%** | **54,798%** |

### 8.3. Hiệu năng model-only

Gate latency chung đo batch 1, audio đã preload, 10 warm-up và 50 lần đo bằng CUDA events trên Tesla T4:

| Model | Median latency | p95 latency | Parameters |
| :---- | ----: | ----: | ----: |
| ECAPA-TDNN | 12,023 ms | 12,912 ms | 20,77 M |
| RawNet3 | 8,840 ms | 12,884 ms | 16,28 M |

Số này không bao gồm đọc file, resample, app, network hoặc ASR/TTS. Không có benchmark CPU nên không được suy diễn latency CPU.

## 9\. So sánh và lựa chọn

### TidyVoice

ECAPA giảm EER tương đối 33,4%, giảm minDCF tương đối 36,2% và tăng TAR tại ngưỡng đóng băng 17,97 điểm phần trăm so với RawNet3. ECAPA là model tốt hơn trên TidyVoice.

### ViMD

RawNet3 giảm EER tương đối 19,3%, giảm minDCF tương đối 49,9% và tăng TAR tại ngưỡng đóng băng 19,19 điểm phần trăm so với ECAPA. RawNet3 còn ít hơn 21,6% tham số và nhanh hơn 26,5% trong latency gate batch 1\.

### Quyết định

Chọn **RawNet3/ViMD best epoch 2** bằng evidence Validation và ràng buộc triển khai; Test chỉ xác nhận kết quả cuối. Lý do:

1. Ứng dụng mục tiêu nhận giọng nói người Việt; ViMD gần target domain hơn.  
2. Trên ViMD Validation, RawNet3 có EER `2,692%` và minDCF `0,2446`, tốt hơn ECAPA với EER `3,936%` và minDCF `0,4503`.  
3. Model nhỏ hơn và nhanh hơn ECAPA trong phép đo T4 đã khai báo.  
4. Training RawNet3/ViMD cải thiện liên tục qua ba epoch và artifact đầy đủ.  
5. ViMD Test sau đó xác nhận cùng xu hướng về EER, minDCF và TAR bảo mật.

Không kết luận RawNet3 tốt hơn tuyệt đối: ECAPA thắng rõ trên TidyVoice. Quyết định là lựa chọn theo domain và yêu cầu triển khai, không phải xếp hạng phổ quát.

## 10\. Tính toàn vẹn của evidence

Hai ZIP đã qua CRC, path-safety, kiểm tra checkpoint SHA-256 và validator chính thức cho cả bốn run; không có NaN/Infinity hoặc traceback trong run được nhận.	

| Archive | SHA-256 |
| :---- | :---- |
| ECAPA | `519e431c751287f2b891978366d17d93877b72063546b4a307ce38f49984faa3` |
| RawNet3 | `593f416cbf68025f09d6415b6d00bdc9f444382e3c1d9396f1426951b9658253` |

## 11\. Hạn chế phải công bố

- Chỉ dùng seed 42; chưa đo variance giữa nhiều seed.  
- Tối đa ba epoch, một utterance/speaker/epoch và patience 1 có thể underfit hoặc dừng do nhiễu Validation.  
- Không có grid search, scheduler ablation, margin ablation hoặc augmentation study; không gọi config hiện tại là tối ưu.  
- Test trials dùng chung speaker/utterance nên các trial không độc lập hoàn toàn.  
- Threshold được hiệu chuẩn trên ViMD, chưa hiệu chuẩn trên microphone/noise của ứng dụng thật.  
- Latency chỉ là model-only trên Tesla T4; chưa phải end-to-end latency.  
- WavLM+MHFA là thử nghiệm bị loại, không phải kết quả hoàn thành.

## 12\. Ưu và nhược điểm của phương án

Ưu điểm:

- So sánh công bằng bằng cùng split, preprocessing, objective, trial và metrics.  
- Không leakage Test trong early stopping, checkpoint hoặc threshold selection.  
- Dùng pretrained source đã pin revision, restricted loading và artifact hash.  
- Bao phủ toàn bộ speaker Train dù giới hạn một utterance/speaker/epoch.  
- Có checkpoint và evidence đủ để bàn giao.

Nhược điểm:

- Compute-constrained làm giảm độ mạnh của kết luận.  
- Crop duration theo kiến trúc khác nhau, dù có lý do từ source compatibility.  
- Hai dataset cho ranking khác nhau; cần chọn theo target domain.  
- Checkpoint training chứa cả optimizer/head nên lớn hơn artifact inference tối thiểu.

# Hướng dẫn tích hợp model vào app

# Hướng dẫn bàn giao và tích hợp RawNet3 vào ứng dụng

## 1\. Artifact phải dùng

Model được chọn là **RawNet3 fine-tune trên ViMD, best Validation epoch 2** (epoch thứ ba vì artifact đánh index từ 0).

Lấy đúng file sau từ ZIP:

ZIP: [https://drive.google.com/file/d/1GvvGbe1bOwXTG2fj7BziDuyQfBYlZ409/view?usp=sharing](https://drive.google.com/file/d/1GvvGbe1bOwXTG2fj7BziDuyQfBYlZ409/view?usp=sharing) 

model/Thanh2/results/team\_runs/who\_speak\_ai\_rawnet3\_resource\_constrained.zip

Checkpoint bên trong ZIP:

rawnet3/runs/vimd/checkpoints/best.pt

Checkpoint sidecar:

rawnet3/runs/vimd/checkpoints/best.pt.json

Resolved config:

rawnet3/runs/vimd/resolved\_config.json

Final evidence:

rawnet3/runs/vimd/final\_test.json

SHA-256 bắt buộc:

ZIP:

593f416cbf68025f09d6415b6d00bdc9f444382e3c1d9396f1426951b9658253

best.pt:

0b06fd3c4644d6c1cf4e7f9c087cf7fba8be493952589ab4321f8319d4215386

Không dùng:

- `last.pt` thay cho `best.pt`;  
- RawNet3 pretrained `model.pt` thay cho checkpoint đã fine-tune;  
- checkpoint TidyVoice;  
- AAM classifier head để xác thực user trong app;  
- threshold ví dụ `0.75` trong tài liệu concept cũ.

## 2\. Giải nén và kiểm tra trên PowerShell

Chạy từ repository root:

$zip \= ".\\model\\Thanh2\\results\\team\_runs\\who\_speak\_ai\_rawnet3\_resource\_constrained.zip"

$destination \= ".\\model\\Thanh2\\results\\selected\_model"

Get-FileHash \-Algorithm SHA256 $zip

Expand-Archive \-LiteralPath $zip \-DestinationPath $destination

$checkpoint \= Join-Path $destination "rawnet3\\runs\\vimd\\checkpoints\\best.pt"

Get-FileHash \-Algorithm SHA256 $checkpoint

Chỉ tiếp tục khi hai hash khớp mục 1\. Giữ `best.pt.json`, `resolved_config.json` và `final_test.json` cùng artifact bàn giao để truy vết nguồn gốc.

## 3\. Source và môi trường

Run RawNet3 được tạo bằng code tại revision:

c68471a69c089cc40a5975b22362da37abcac186

Thông tin runtime đã xác thực:

- Python 3.12 trên Kaggle;  
- PyTorch `2.10.0+cu128`;  
- CUDA build 12.8;  
- `asteroid-filterbanks==0.4.0`;  
- `huggingface-hub>=1.11,<1.12`;  
- model checkpoint source `jungjee/RawNet3`, revision `c89102eea20c3f96917c434de673c0ace0caddc0`;  
- architecture source `clovaai/voxceleb_trainer`, revision `f51bab870672a9b0b50fa158b4e30f329e7866d7`.

PyTorch không nằm trong dependency group vì phải cài build phù hợp CPU/CUDA của máy triển khai trước. Sau đó, tại `model/Thanh2`:

python \-m pip install \-e ".\[data,rawnet3\]"

Nếu dùng CPU, model vẫn có thể chạy nhưng dự án chưa benchmark latency CPU và threshold hiện tại được đo bằng CUDA FP16. Phải hiệu chuẩn lại threshold bằng dữ liệu app trước khi gọi là production-ready.

## 4\. Contract inference bắt buộc

| Thành phần | Giá trị |
| :---- | :---- |
| Sample rate | 16.000 Hz |
| Channels | Mono; nhiều kênh lấy trung bình số học |
| Resampling | SciPy polyphase |
| Amplitude normalization | Không dùng |
| Evaluation segment | 64.240 mẫu \= 4,015 giây |
| Segment count hiện tại | 1 |
| Short audio | Lặp waveform đến đủ độ dài |
| Embedding | 256 chiều, L2-normalized |
| Score | Cosine similarity |
| Accept rule | `score >= threshold` |
| SV threshold ban đầu | `0.6565317440180312` |

Không thay đổi preprocessing nhưng giữ nguyên threshold, vì như vậy score distribution không còn tương ứng với Validation đã hiệu chuẩn.

## 5\. Loader inference tham khảo

Đoạn mã sau dùng đúng adapter, preprocessing, crop và checkpoint structure của training pipeline. Đội app nên đưa class này vào service/model layer, không đặt trực tiếp trong UI callback.

"""Load the selected RawNet3 checkpoint and produce speaker embeddings."""

from \_\_future\_\_ import annotations

import hashlib

from collections.abc import Mapping, Sequence

from pathlib import Path

import numpy as np

import soundfile as sf

import torch

import torch.nn.functional as functional

from speaker\_recognition.data.audio import canonicalize\_audio

from speaker\_recognition.data.segments import evenly\_spaced\_segments

from speaker\_recognition.models.rawnet3 import (

    RAWNET3\_EVALUATION\_SAMPLES,

    RawNet3Adapter,

)

SELECTED\_CHECKPOINT\_SHA256 \= (

    "0b06fd3c4644d6c1cf4e7f9c087cf7fba8be493952589ab4321f8319d4215386"

)

SELECTED\_CONFIG\_SHA256 \= (

    "08392e8146fb4ae42cc7b971641d4339c9f98f517bd6f3ece1d5b2bf2fdafa94"

)

SV\_THRESHOLD \= 0.6565317440180312

def sha256\_file(path: Path, chunk\_bytes: int \= 1024 \* 1024\) \-\> str:

    """Hash a model file incrementally so a corrupted artifact is rejected."""

    digest \= hashlib.sha256()

    with path.open("rb") as stream:

        while chunk := stream.read(chunk\_bytes):

            digest.update(chunk)

    return digest.hexdigest()

class RawNet3SpeakerEncoder:

    """Wrap the selected fine-tuned encoder behind one inference contract."""

    def \_\_init\_\_(

        self,

        checkpoint\_path: str | Path,

        \*,

        cache\_dir: str | Path,

        device: str \= "cuda:0",

    ) \-\> None:

        """Authenticate, safely load, and freeze the selected RawNet3 model."""

        self.device \= torch.device(device)

        self.checkpoint\_path \= Path(checkpoint\_path).expanduser().resolve()

        if sha256\_file(self.checkpoint\_path) \!= SELECTED\_CHECKPOINT\_SHA256:

            raise RuntimeError("Selected RawNet3 checkpoint SHA-256 mismatch.")

        \# This constructs the exact pinned architecture. The source checkpoint

        \# is authenticated by RawNet3Adapter before fine-tuned weights replace it.

        self.adapter \= RawNet3Adapter.from\_pretrained(

            cache\_dir=cache\_dir,

            device=str(self.device),

        )

        payload \= torch.load(

            self.checkpoint\_path,

            map\_location="cpu",

            weights\_only=True,

        )

        if not isinstance(payload, Mapping):

            raise RuntimeError("Training checkpoint root must be a mapping.")

        identity \= payload.get("identity")

        expected\_identity \= {

            "model\_name": "rawnet3",

            "dataset\_name": "vimd",

            "config\_sha256": SELECTED\_CONFIG\_SHA256,

            "manifest\_sha256": (

                "ed7b764c6aaab2ba2c4ec95edadab19fd640ebca72aa06da3d36cbf93fc4747f"

            ),

            "seed": 42,

        }

        if identity \!= expected\_identity:

            raise RuntimeError("Checkpoint experiment identity mismatch.")

        adapter\_state \= payload.get("adapter\_state")

        if not isinstance(adapter\_state, Mapping):

            raise RuntimeError("Checkpoint has no adapter state mapping.")

        self.adapter.load\_state\_dict(adapter\_state, strict=True)

        self.adapter.to(self.device)

        self.adapter.eval()

        self.adapter.requires\_grad\_(False)

    def embed\_file(self, audio\_path: str | Path) \-\> np.ndarray:

        """Return one normalized 256-D embedding from an audio file."""

        waveform, sample\_rate \= sf.read(

            Path(audio\_path),

            dtype="float32",

            always\_2d=True,

        )

        canonical \= canonicalize\_audio(

            waveform,

            sample\_rate=int(sample\_rate),

            target\_sample\_rate=16000,

        )

        segments \= evenly\_spaced\_segments(

            canonical,

            num\_samples=RAWNET3\_EVALUATION\_SAMPLES,

            segment\_count=1,

        )

        batch \= torch.from\_numpy(segments).to(self.device)

        with torch.inference\_mode():

            if self.device.type \== "cuda":

                \# This matches the accepted evaluation precision.

                with torch.autocast(

                    device\_type="cuda",

                    dtype=torch.float16,

                    enabled=True,

                ):

                    crop\_embeddings \= self.adapter(batch)

            else:

                crop\_embeddings \= self.adapter(batch)

            utterance\_embedding \= functional.normalize(

                crop\_embeddings.float().mean(dim=0, keepdim=True),

                p=2,

                dim=1,

            )

        return utterance\_embedding.squeeze(0).cpu().numpy()

def average\_enrollment(embeddings: Sequence\[np.ndarray\]) \-\> np.ndarray:

    """Average multiple recordings and L2-normalize one user template."""

    if len(embeddings) \< 3:

        raise ValueError("Enrollment requires at least three recordings.")

    matrix \= np.stack(embeddings).astype(np.float32, copy=False)

    template \= matrix.mean(axis=0)

    norm \= float(np.linalg.norm(template))

    if not np.isfinite(norm) or norm \<= 0.0:

        raise ValueError("Enrollment template has invalid norm.")

    return np.ascontiguousarray(template / norm, dtype=np.float32)

def cosine\_score(query: np.ndarray, template: np.ndarray) \-\> float:

    """Score two normalized embeddings using their dot product."""

    score \= float(np.dot(query, template))

    if not np.isfinite(score):

        raise ValueError("Cosine score is not finite.")

    return score

## 6\. Enrollment

Quy trình tối thiểu cho một user:

1. Thu ba recording riêng, mỗi recording nên chứa speech rõ và dài ít nhất khoảng 4,015 giây để tránh phải lặp audio.  
2. Kiểm tra file decode được, không rỗng và không có sample không hữu hạn.  
3. Chạy `embed_file()` cho từng recording.  
4. Lấy trung bình ba embedding rồi L2-normalize bằng `average_enrollment()`.  
5. Lưu template 256 chiều cùng metadata:  
   - `user_id`;  
   - tên hiển thị;  
   - model `rawnet3-vimd-best-epoch-2`;  
   - checkpoint SHA-256;  
   - preprocessing version;  
   - thời điểm enrollment.  
6. Không lưu raw audio nếu use case không cần; nếu lưu phải có đồng ý của user, phân quyền truy cập và chính sách xóa.

JSON có thể lưu embedding dạng list cho demo. Với hệ thống lớn hơn, dùng SQLite và BLOB/array storage. Không dùng AAM class index làm user ID: head training chỉ phục vụ học embedding và không đại diện cho user mới của app.

## 7\. Speaker Verification 1:1

query \= encoder.embed\_file("query.wav")

score \= cosine\_score(query, enrolled\_template)

verified \= score \>= SV\_THRESHOLD

Ngưỡng `0.6565317440180312` được chọn trên ViMD Validation tại FAR mục tiêu 0,1%, sau đó đóng băng. Trên ViMD Test, ngưỡng này đạt:

- FAR 0,075%;  
- FRR 15,547%;  
- TAR 84,453%.

Đây là ngưỡng khởi tạo có evidence, không phải đảm bảo production. Microphone, nhiễu, codec, passphrase và nhóm user của app có thể làm score distribution thay đổi. Khi có dữ liệu app, chọn lại threshold trên một app-validation set; không chỉnh bằng app-test hoặc vài case demo mong muốn.

## 8\. Speaker Identification 1:N

query embedding

  \-\> cosine với mọi enrollment template

  \-\> chọn score lớn nhất

  \-\> nếu score \>= SID unknown threshold: trả user tương ứng

  \-\> ngược lại: guest/unknown

Threshold hiện tại được hiệu chuẩn cho SV 1:1, chưa được xác thực cho open-set SID 1:N. Có thể dùng nó làm giá trị demo ban đầu nhưng phải ghi là provisional. Threshold SID cần được chọn riêng trên dữ liệu validation gồm known và unknown speakers của app; khi số user tăng, xác suất false identification cũng thay đổi.

## 9\. Gắn vào luồng ứng dụng

Audio input

  \-\> canonical preprocessing

  \-\> RawNet3 embedding

  \-\> SID 1:N

       \-\> known: lấy profile, cá nhân hóa prompt

       \-\> unknown: guest mode

  \-\> intent/router

       \-\> tác vụ thường: thực thi

       \-\> nhật ký cá nhân: SV 1:1 với claimed/identified user

            \-\> pass: thực thi

            \-\> fail: từ chối

Model nên được load đúng một lần khi service khởi động. Không load checkpoint lại sau mỗi request. Enrollment templates có thể cache trong RAM nhưng database vẫn là nguồn dữ liệu chính.

## 10\. Acceptance checklist cho đội app

- [ ] Hash ZIP và `best.pt` khớp.  
- [ ] Dùng RawNet3/ViMD `best.pt`, không dùng `last.pt` hoặc base `model.pt`.  
- [ ] Restricted load với `weights_only=True` và `strict=True`.  
- [ ] Model ở `eval()` và inference dùng `torch.inference_mode()`.  
- [ ] Preprocessing đúng mono float32 16 kHz, không amplitude normalization.  
- [ ] Embedding cuối có shape `(256,)`, hữu hạn và norm xấp xỉ 1\.  
- [ ] Enrollment trung bình tối thiểu ba recording rồi normalize lại.  
- [ ] Cosine score hữu hạn; accept rule là `>=`.  
- [ ] SV threshold ban đầu đúng `0.6565317440180312`.  
- [ ] SID có nhánh unknown/guest; không ép mọi giọng nói vào một user.  
- [ ] Model chỉ load một lần.  
- [ ] Có test same-speaker, different-speaker, unknown, audio ngắn, stereo và sample rate khác 16 kHz.  
- [ ] Có đo end-to-end latency trên máy thật; không dùng latency T4 làm latency app.  
- [ ] Không dùng dữ liệu Test để điều chỉnh threshold.

## 11\. Ưu và nhược điểm của lựa chọn

Ưu điểm:

- Tốt nhất trên ViMD trong bốn run được chấp nhận.  
- 16,28 triệu tham số, ít hơn ECAPA 21,6%.  
- Batch-1 model-only median 8,840 ms trên T4, nhanh hơn ECAPA trong cùng gate.  
- Một embedding dùng được cho cả SV và SID.  
- Checkpoint, config và threshold có hash/provenance rõ ràng.

Nhược điểm:

- Checkpoint training khoảng 227 MB vì chứa optimizer và head, lớn hơn encoder inference tối thiểu.  
- Chưa benchmark CPU, mobile hoặc end-to-end app.  
- Threshold chưa hiệu chuẩn trên microphone và user population của app.  
- RawNet3 kém ECAPA trên TidyVoice, nên lựa chọn phụ thuộc target tiếng Việt.

# Cách seminar với thầy phần train model

# Kịch bản báo cáo và vấn đáp phần speaker model trong 10 phút

## 1\. Thông điệp phải giữ nhất quán

> Nhóm xây dựng một pipeline chung để fine-tune và đánh giá ECAPA-TDNN và RawNet3 trên TidyVoice và ViMD. Bốn run dùng cùng split policy, preprocessing, objective, trial protocol và metric code. ECAPA tốt hơn trên TidyVoice; RawNet3 tốt hơn trên ViMD, nhỏ hơn và nhanh hơn trên T4. Vì app hướng đến tiếng Việt, nhóm chọn RawNet3/ViMD best epoch 2 bằng Validation; Test xác nhận kết quả cuối. Kết quả bị giới hạn bởi ba epoch tối đa và một seed nên không được gọi là tối ưu.

Không nói:

- “RawNet3 luôn tốt hơn ECAPA.”  
- “Đây là config tối ưu.”  
- “Accuracy 99,76% nghĩa là model gần như hoàn hảo.”  
- “Nhóm đã hoàn thành WavLM.”  
- “Test được dùng để chọn epoch hoặc threshold.”

## 2\. Chuẩn bị trước khi trình bày

Mở sẵn các file/tab sau theo thứ tự:

1. `docs/report/train_eval_models.md` — bảng kết quả chính.  
2. `configs/base.toml` — shared controls.  
3. `configs/stages/resource_constrained.toml` — giới hạn compute.  
4. `configs/models/ecapa_tdnn.toml` và `configs/models/rawnet3.toml` — phần riêng theo kiến trúc.  
5. `src/speaker_recognition/data/audio.py`, hàm `canonicalize_audio()`.  
6. `src/speaker_recognition/training/objectives.py`, class `AamSoftmaxHead`.  
7. `src/speaker_recognition/training/engine.py`, class `SpeakerTrainingEngine`.  
8. `src/speaker_recognition/evaluation/metrics.py`, hàm `compute_verification_metrics()`.  
9. `scripts/validate_training_run.py`, hàm `validate_training_run()`.  
10. `docs/report/app_intergrate.md` — checkpoint và threshold bàn giao.

Nếu cần show evidence gốc, giải nén trước bốn file sau từ hai ZIP:

\<model\>/runs/\<dataset\>/resolved\_config.json

\<model\>/runs/\<dataset\>/run\_summary.json

\<model\>/runs/\<dataset\>/final\_test.json

\<model\>/runs/\<dataset\>/checkpoints/best.pt.json

Không mở checkpoint binary trong editor và không chạy training trong buổi báo cáo.

## 3\. Flow trình bày 10 phút

### 0:00–0:45 — Bài toán

Nói:

> Phần của em là speaker recognition cho trợ lý ảo bảo mật. Model phải tạo embedding để làm hai việc: verification 1:1 cho tác vụ nhạy cảm và identification 1:N để cá nhân hóa. Yêu cầu tối thiểu là một model; nhóm em huấn luyện và đánh giá hai model hoàn chỉnh để có cơ sở lựa chọn.

Minh họa:

audio \-\> speaker embedding \-\> cosine score

                         \-\> 1:1 verification

                         \-\> 1:N identification

### 0:45–2:00 — Dataset và chống leakage

Nói:

> Nhóm dùng TidyVoice và ViMD. TidyVoice có 262.268 utterance Train, còn Dev được chia speaker-disjoint thành Validation 29.720 và Test 29.723 utterance. ViMD có 15.023 Train, 1.898 Validation và 2.026 Test utterance. Audit phát hiện hai speaker trùng giữa Validation và Test của ViMD, nên nhóm loại đúng hai singleton khỏi Validation và giữ Test nguyên trạng. Không có speaker leakage giữa các split chuẩn hóa.

Nói thêm nếu còn thời gian:

- TidyVoice Dev: 808 speaker → 404 Validation \+ 404 Test.  
- ViMD: Train 10.291 speaker; Validation 1.318; Test 1.344.  
- Hai hàng ViMD bị loại không làm giảm genuine-pair capacity Validation.

Show:

- `docs/decisions/001_tidyvoice_dev_protocol.md`.  
- `docs/decisions/002_vimd_canonical_protocol.md`.

### 2:00–3:10 — Preprocessing và hai kiến trúc

Nói:

> Hai dataset có format khác nhau nhưng mọi model nhận cùng mono float32 16 kHz. Stereo được lấy trung bình kênh, resample bằng polyphase, không amplitude normalization. TidyVoice đọc WAV; ViMD đọc audio bytes trong Parquet theo row-group cache.  
>   
> ECAPA-TDNN dùng pretrained SpeechBrain, 20,77 triệu tham số và embedding 192 chiều. RawNet3 xử lý waveform trực tiếp bằng Sinc filterbank, residual encoder và ECA pooling, có 16,28 triệu tham số và embedding 256 chiều. Cả hai bỏ source classifier và tạo head mới theo speaker của target dataset.

Show code:

- `data/audio.py:189`, `canonicalize_audio()` — downmix/resample/finiteness.  
- `models/ecapa_tdnn.py:119`, `EcapaTdnnAdapter`.  
- `models/rawnet3.py:46`, `RawNet3Adapter`.  
- `models/rawnet3.py:79`, checkpoint hash \+ restricted strict load.

### 3:10–5:10 — Thiết kế training và lý do chọn parameter

Nói:

> Để so sánh công bằng, hai model dùng cùng seed 42, AAM-Softmax margin 0,2, scale 30, FP16, loss scale 1024, gradient clipping 5, constant learning rate, cùng trial và metric code. Khác biệt chỉ giữ ở nơi kiến trúc bắt buộc.  
>   
> ECAPA và RawNet3 đều fine-tune toàn encoder cùng head mới bằng Adam, learning rate 1e-4. Weight decay lần lượt là 2e-6 và 5e-5 theo recipe/source tương ứng. Đây là source-informed hypotheses, không phải hyperparameter tối ưu.  
>   
> Batch size được chọn bằng evidence: cả hai pass đến 32 với head ViMD 10.291 class. Nhóm lấy batch 24 để giữ khoảng an toàn, sau đó pass thêm ba optimizer step trên ba batch audio thật. Vì chỉ còn khoảng tám giờ, stage cuối dùng mọi speaker nhưng một utterance/speaker/epoch, tối đa ba epoch và patience 1\.

Show:

- `configs/base.toml`: seed, audio, AAM, FP16, metrics.  
- `configs/stages/resource_constrained.toml`: một utterance/speaker, ba epoch, patience 1\.  
- `configs/models/*.toml`: batch, crop, optimizer, LR, weight decay.  
- `training/objectives.py:26`, `AamSoftmaxHead`: float32 angular math trong mixed precision và không tạo dense one-hot.  
- `training/engine.py:108`, `SpeakerTrainingEngine`: train, early stop, checkpoint.

Nếu thầy hỏi vì sao không dùng chung hoàn toàn mọi parameter:

> So sánh công bằng không có nghĩa ép kiến trúc khác nhau nhận parameter không phù hợp. Shared controls được giữ giống nhau; crop và source recipe khác nhau được khai báo riêng, có provenance và gate thực nghiệm.

### 5:10–6:10 — Cách đánh giá

Nói:

> Mỗi split có 100.000 impostor trials để đo FAR đến vùng 0,01%; genuine pairs được cap 20/speaker. Score là cosine giữa embedding L2-normalized. Chỉ Validation EER chọn best checkpoint; minDCF tie-break. Threshold bảo mật được chọn trên Validation tại FAR mục tiêu 0,1%, đóng băng rồi áp dụng lên Test. Test không tham gia training, early stopping hoặc tuning threshold.

Định nghĩa ngắn:

- FAR: tỷ lệ người sai được chấp nhận; thấp hơn tốt hơn cho bảo mật.  
- FRR: tỷ lệ người đúng bị từ chối.  
- TAR \= `1 - FRR`.  
- EER: điểm FAR bằng FRR; thấp hơn tốt hơn.  
- minDCF: chi phí phát hiện nhỏ nhất với prior/cost đã khai báo; thấp hơn tốt hơn.

Show:

- `evaluation/metrics.py:279`, `compute_verification_metrics()`.  
- `docs/decisions/005_verification_trial_protocol.md`.

### 6:10–7:45 — Kết quả

Show bảng này và đọc số chính, không đọc mọi cột:

| Model | Dataset | EER ↓ | minDCF ↓ | FAR đóng băng | TAR đóng băng |
| :---- | :---- | ----: | ----: | ----: | ----: |
| ECAPA | TidyVoice | **3,912%** | **0,3586** | 0,080% | **71,841%** |
| RawNet3 | TidyVoice | 5,875% | 0,5624 | 0,124% | 53,874% |
| ECAPA | ViMD | 3,804% | 0,4516 | 0,116% | 65,259% |
| RawNet3 | ViMD | **3,071%** | **0,2262** | **0,075%** | **84,453%** |

Nói:

> ECAPA thắng trên TidyVoice. RawNet3 thắng trên ViMD: EER thấp hơn 19,3%, minDCF thấp hơn 49,9% tương đối và TAR tại ngưỡng đóng băng cao hơn 19,19 điểm phần trăm so với ECAPA. Accuracy không phải chỉ số chính vì protocol có 100.000 impostor nhưng ít genuine hơn.

Diễn biến đáng chú ý:

- ECAPA/TidyVoice, ECAPA/ViMD và RawNet3/TidyVoice chọn epoch 0 rồi dừng sớm sau epoch 1 xấu hơn.  
- RawNet3/ViMD cải thiện Validation EER `29,664% -> 7,532% -> 2,692%`, chọn epoch 2\.

### 7:45–8:40 — Vì sao chọn RawNet3/ViMD

Nói nguyên ý sau:

> Nhóm không chọn checkpoint hoặc kiến trúc bằng Test. Checkpoint được chọn bằng Validation EER. Trên ViMD Validation, RawNet3 đạt EER 2,692% và minDCF 0,2446, tốt hơn ECAPA 3,936% và 0,4503. App phục vụ tiếng Việt nên ViMD là domain ưu tiên. ViMD Test sau đó xác nhận RawNet3 tốt hơn về EER, minDCF và TAR bảo mật. RawNet3 có 16,28 triệu tham số, ít hơn ECAPA 21,6%; batch-1 model-only median 8,840 ms trên T4, nhanh hơn ECAPA 12,023 ms. Vì vậy chọn RawNet3/ViMD best epoch 2\.

Checkpoint:

rawnet3/runs/vimd/checkpoints/best.pt

SHA-256: 0b06fd3c4644d6c1cf4e7f9c087cf7fba8be493952589ab4321f8319d4215386

SV threshold: 0.6565317440180312

Nhắc nếu thầy hỏi: epoch 2 là epoch thứ ba vì index bắt đầu từ 0\.

### 8:40–9:20 — WavLM và hạn chế

Nói:

> Nhóm có triển khai và chạy WavLM+MHFA nhưng không đưa vào kết quả vì không có run hoàn chỉnh: một lần lỗi non-finite embedding trong training, lần sau lỗi non-finite evidence ở ViMD Final Test. Nhóm loại thay vì báo cáo metric chưa xác thực. Hai model hoàn chỉnh vẫn vượt yêu cầu tối thiểu một model.  
>   
> Hạn chế chính: một seed, tối đa ba epoch, một utterance/speaker/epoch, patience 1, chưa có hyperparameter search, chưa benchmark CPU và chưa hiệu chuẩn threshold trên microphone app. Vì vậy kết luận là compute-constrained và theo domain, không phải tối ưu phổ quát.

### 9:20–10:00 — Bàn giao app

Nói:

> Đội app lấy đúng `best.pt`, kiểm tra SHA-256, load `adapter_state` bằng `weights_only=True` và `strict=True`, giữ preprocessing mono 16 kHz và crop 64.240 mẫu. Enrollment lấy ít nhất ba recording, trung bình embedding rồi normalize. SV dùng cosine và threshold Validation 0,65653 làm mức ban đầu. SID dùng nearest cosine nhưng unknown threshold phải hiệu chuẩn riêng trên dữ liệu app. Tài liệu đầy đủ nằm trong `docs/report/app_intergrate.md`.

## 4\. Flow show code trong 90 giây

Nếu thầy yêu cầu xem code, đi theo flow này:

1\. configs/base.toml

   \-\> chứng minh shared controls, seed, metrics, AAM, FP16

2\. configs/models/rawnet3.toml

   \-\> provenance, parameter count, batch/crop/LR

3\. data/audio.py::canonicalize\_audio

   \-\> một preprocessing contract cho hai dataset

4\. models/rawnet3.py::from\_pretrained

   \-\> pin revision, SHA-256, weights\_only, strict load

5\. training/objectives.py::AamSoftmaxHead.forward

   \-\> objective và numerical safeguard

6\. training/engine.py::fit / \_train\_batch

   \-\> lifecycle, AMP, clipping, checkpoint, early stopping

7\. evaluation/metrics.py::compute\_verification\_metrics

   \-\> EER, minDCF, FAR/FRR/TAR

8\. scripts/validate\_training\_run.py::validate\_training\_run

   \-\> evidence và checkpoint hash được kiểm định

9\. final\_test.json

   \-\> best Validation checkpoint \+ frozen threshold \+ Test metrics

## 5\. Câu hỏi vấn đáp thường gặp

### Tại sao dùng pretrained rồi vẫn gọi là train?

Fine-tune toàn encoder trên target speaker labels bằng head AAM-Softmax mới. Đây không phải chỉ chạy inference pretrained. Pretraining giúp hội tụ nhanh và phù hợp giới hạn compute.

### Tại sao dùng AAM-Softmax?

AAM-Softmax thêm angular margin giữa các speaker trên hypersphere, phù hợp với embedding L2-normalized và cosine scoring. Margin 0,2 và scale 30 là shared source-informed control; chưa có ablation nên không gọi là tối ưu.

### Tại sao batch 24?

Memory calibration pass đến 32 với classifier ViMD 10.291 class. Chọn 24 để có headroom, rồi xác nhận bằng ba batch audio thật và ba optimizer updates hữu hạn.

### Tại sao chỉ một utterance/speaker/epoch?

Giới hạn thời gian khoảng tám giờ. Chính sách này vẫn bao phủ mọi speaker và xoay utterance xác định theo epoch, đồng thời giảm thiên lệch speaker nhiều dữ liệu. Nhược điểm là có thể underfit.

### Tại sao không train đủ 15 epoch?

Kế hoạch 15 epoch không khả thi trong thời gian còn lại. Stage ba epoch được khai báo trước khi có kết quả, không cắt sau khi nhìn Test. Báo cáo công khai giới hạn này.

### Tại sao epoch 0 lại là best ở ba run?

Training loss tiếp tục giảm nhưng Validation EER xấu hơn, dấu hiệu target classification loss không đồng nhất với verification generalization hoặc bắt đầu overfit. Early stopping dùng Validation đã giữ checkpoint tốt hơn.

### Tại sao RawNet3/ViMD epoch 0 rất xấu rồi cải thiện mạnh?

Evidence cho thấy EER giảm liên tục qua ba epoch; không có dữ liệu để khẳng định nguyên nhân sâu hơn. Có thể nói model cần thích nghi target domain, nhưng phải gọi đây là diễn giải hợp lý, không phải kết luận nhân quả.

### Tại sao chọn RawNet3 khi ECAPA tốt hơn trên TidyVoice?

Không có model thắng mọi domain. Target app là tiếng Việt, nên ưu tiên ViMD. RawNet3 tốt hơn cả EER, minDCF và TAR bảo mật trên ViMD, đồng thời nhỏ và nhanh hơn trong T4 gate. ECAPA/TidyVoice được giữ làm bằng chứng trade-off.

### EER và threshold triển khai có giống nhau không?

Không. EER là metric mô tả tại điểm FAR≈FRR. Threshold triển khai được chọn riêng trên Validation tại FAR mục tiêu 0,1%, rồi đóng băng cho Test.

### Tại sao FAR Test không đúng chính xác 0,1%?

Threshold được chọn trên Validation; phân bố Test khác nên observed FAR có thể khác. RawNet3/ViMD đạt 0,075% trên Test. Không được chỉnh threshold bằng Test để ép FAR đúng 0,1%.

### Accuracy 99,765% có đáng tin không?

Con số đúng theo protocol nhưng dễ gây hiểu nhầm vì Test ViMD có 100.000 impostor và 1.042 genuine trials. Accuracy bị chi phối bởi impostor. EER, minDCF và TAR tại FAR thấp quan trọng hơn.

### Threshold 0,65653 có dùng thẳng trong app được không?

Dùng làm baseline có evidence. Trước production phải hiệu chuẩn trên app-validation vì microphone, noise, codec, passphrase và user population khác ViMD. Không dùng app-test để tuning.

### Một embedding làm cả SV và SID được không?

Có. SV so cosine 1:1; SID tìm cosine lớn nhất 1:N. Tuy nhiên unknown threshold cho SID phải hiệu chuẩn riêng vì open-set 1:N khác SV.

### Vì sao WavLM bị loại?

Không có artifact hoàn chỉnh, hữu hạn và qua validator. Báo cáo kết quả chưa được xác thực sẽ kém trung thực hơn việc công khai loại run.

### Có thể tái lập kết quả không?

Có ở mức cấu hình: seed, split/trial/config fingerprint, pinned model revision, checkpoint hash, deterministic crop, strict algorithm settings và complete evidence đều được lưu. Tuy nhiên hạ tầng và single-seed vẫn là giới hạn.

## 6\. Checklist trước khi gặp thầy

- [ ] Thuộc bốn số EER: 3,912%; 5,875%; 3,804%; 3,071%.  
- [ ] Thuộc lý do chọn RawNet3/ViMD.  
- [ ] Phân biệt EER với threshold bảo mật.  
- [ ] Không dùng accuracy làm kết luận chính.  
- [ ] Nhắc rõ Validation chọn checkpoint và threshold; Test chỉ đánh giá cuối.  
- [ ] Nói đúng “tối đa ba epoch, một utterance/speaker/epoch, seed 42”.  
- [ ] Nói “source-informed, resource-constrained”, không nói “optimal”.  
- [ ] Công khai WavLM bị loại và giới hạn single-seed.  
- [ ] Có sẵn checkpoint hash và đường dẫn bàn giao.  
- [ ] Không demo bằng cách load file checkpoint lạ hoặc dùng `weights_only=False`.

## 7\. Ưu và nhược điểm của cách trình bày

Ưu điểm:

- Đi từ bài toán → phương pháp → evidence → lựa chọn → bàn giao.  
- Mọi claim chính đều trỏ được tới code, config hoặc artifact.  
- Chủ động công khai giới hạn nên tránh bị bắt lỗi “tối ưu hóa quá mức”.

Nhược điểm:

- Mười phút không đủ show toàn bộ pipeline; phải giữ đúng flow 90 giây khi mở code.  
- Nhiều metric dễ gây quá tải; chỉ nhấn EER, minDCF, FAR/TAR ở ngưỡng đóng băng.

