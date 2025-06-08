# HỆ THỐNG ĐÁNH GIÁ TỰ ĐỘNG IELTS WRITING TASK 2 SỬ DỤNG REINFORCEMENT LEARNING

## Thông tin dự án
- **Tên dự án:** Hệ thống đánh giá tự động IELTS Writing Task 2 sử dụng Reinforcement Learning
- **Tác giả:** Phạm Lê Ngọc Sơn
- **Email:** phamlengocsononline@gmail.com
- **Thời gian thực hiện:** Tháng 12/2023 - Tháng 3/2024
- **Mô tả:** Nghiên cứu và ứng dụng Machine Learning trong việc đánh giá tự động bài viết IELTS Task 2

## Video Demo
[<img src="https://i.ytimg.com/vi/PDk-nbADeZA/maxresdefault.jpg" width="50%">](https://www.youtube.com/watch?v=PDk-nbADeZA "Demo hệ thống đánh giá IELTS tự động")

## Tổng quan dự án

Dự án này tập trung vào việc phát triển một hệ thống đánh giá tự động cho bài viết IELTS Writing Task 2 sử dụng các kỹ thuật Machine Learning tiên tiến, đặc biệt là Large Language Models (LLMs) và Reinforcement Learning. Hệ thống có khả năng đánh giá chính xác chất lượng bài viết IELTS theo 4 tiêu chí chuẩn và cung cấp phản hồi chi tiết cho người học.

## Cấu trúc dự án

```
Automated_IELTS_Evaluation/
├── data/                           # Thư mục chứa dữ liệu
│   ├── train.csv                   # Dữ liệu huấn luyện (46MB, 9000+ mẫu)
│   ├── test.csv                    # Dữ liệu kiểm tra (2.3MB)
│   ├── preference_data.csv         # Dữ liệu preference cho DPO (13MB, 768 mẫu)
│   └── synthetic_data.py           # Script tạo dữ liệu tổng hợp
├── notebook.ipynb                  # Notebook chính - Fine-tuning models
├── Inference_model.ipynb           # Notebook inference và test model
├── README.md                       # Tài liệu hướng dẫn (file này)
└── .git/                          # Git repository
```

## Chi tiết từng thành phần

### 1. Dữ liệu (data/)

#### a) Dataset SFT (Supervised Fine-Tuning)
- **File:** `train.csv`
- **Nội dung:** Bao gồm đề bài, bài viết, điểm band thực tế và đánh giá chi tiết
- **Kích thước:** Hơn 9,000 mẫu dữ liệu
- **Nguồn:** Tất cả bài viết đều từ các kỳ thi IELTS thực tế năm 2022-2023
- **Liên kết HuggingFace:** https://huggingface.co/datasets/chillies/IELTS-writing-task-2-evaluation

#### b) Dataset Preference (DPO)
- **File:** `preference_data.csv`
- **Nội dung:** Bao gồm bài viết, mẫu được chọn và mẫu bị từ chối
- **Kích thước:** 768 mẫu
- **Liên kết HuggingFace:** https://huggingface.co/datasets/chillies/IELTS_essay_human_feedback

#### c) Script tạo dữ liệu tổng hợp
- **File:** `synthetic_data.py`
- **Chức năng:** Sử dụng Gemini API để tạo thêm dữ liệu đánh giá IELTS
- **Công nghệ sử dụNG:** Google Generative AI, Gemini-1.0-Pro

### 2. Notebook chính (notebook.ipynb)

Notebook này chứa toàn bộ quy trình fine-tuning các model:

#### Các bước thực hiện:
1. **Chuẩn bị môi trường:** Cài đặt các thư viện cần thiết (unsloth, xformers, bitsandbytes)
2. **Load dataset:** Từ HuggingFace datasets
3. **Định nghĩa prompt template:** Template đánh giá theo 4 tiêu chí IELTS
4. **Load và chuẩn bị model:** Mistral-7b với 4-bit quantization
5. **Fine-tuning:** Sử dụng kỹ thuật QLoRA
6. **Đánh giá model:** Test performance trên validation set

#### Models được fine-tune:
- **Mistral-7b** (Model chính được chọn)
- **Llama-2-7b**
- **Gemma-7b**

### 3. Notebook inference (Inference_model.ipynb)

Notebook này dùng để test và sử dụng model đã được train:
- Load model đã fine-tune
- Thực hiện inference trên dữ liệu mới
- Đánh giá kết quả và so sánh với ground truth

## Phương pháp và kỹ thuật

### 1. Supervised Fine-Tuning (SFT)
- **Kỹ thuật:** QLoRA (Quantized Low-Rank Adaptation)
- **Hardware:** Google Colab Tesla T4 GPU (miễn phí)
- **Base models:** Mistral-7b, Llama-2-7b, Gemma-7b
- **Kết quả:** Mistral-7b cho hiệu suất tốt nhất

### 2. Direct Preference Optimization (DPO)
- **Mục đích:** Cải thiện chất lượng output dựa trên human feedback
- **Dataset:** 768 mẫu preference data
- **Kết quả:** Model DPO với độ chính xác cao hơn

### 3. Tiêu chí đánh giá IELTS

Hệ thống đánh giá theo 4 tiêu chí chuẩn của IELTS:

#### Task Achievement (Hoàn thành nhiệm vụ)
- Đánh giá mức độ trả lời đúng yêu cầu đề bài
- Tính rõ ràng và liên quan của ý tưởng
- Việc cover đầy đủ các khía cạnh của đề bài

#### Coherence and Cohesion (Tính mạch lạc và liên kết)
- Tổ chức và cấu trúc bài viết
- Sử dụng từ nối và liên kết ý tưởng
- Luồng thông tin logic

#### Lexical Resource (Từ vựng)
- Phạm vi và độ chính xác của từ vựng
- Phát hiện lỗi từ vựng và đề xuất sửa
- Tính phù hợp của từ vựng với ngữ cảnh

#### Grammatical Range and Accuracy (Ngữ pháp)
- Đa dạng và phức tạp của cấu trúc câu
- Phát hiện lỗi ngữ pháp và đề xuất sửa
- Sử dụng dấu câu và cấu tạo câu

## Models đã public

### 1. SFT Model
- **Tên:** IELTS-fighter
- **Liên kết:** https://huggingface.co/chillies/IELTS-fighter
- **Mô tả:** Model sau khi Supervised Fine-tuning

### 2. DPO Model  
- **Tên:** DPO_ielts_fighter
- **Liên kết:** https://huggingface.co/chillies/DPO_ielts_fighter
- **Mô tả:** Model sau khi áp dụng Direct Preference Optimization

## Hướng dẫn sử dụng

### 1. Cài đặt môi trường

```bash
# Clone repository
git clone <repository-url>
cd Automated_IELTS_Evaluation

# Cài đặt các thư viện cần thiết
pip install unsloth[colab]
pip install datasets transformers torch
pip install google-generativeai python-dotenv pandas
```

### 2. Chạy notebook fine-tuning

```bash
# Mở notebook chính
jupyter notebook notebook.ipynb

# Hoặc sử dụng Google Colab (khuyến khích để có GPU miễn phí)
```

### 3. Sử dụng model để đánh giá

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model đã fine-tune
model_name = "chillies/DPO_ielts_fighter"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Định nghĩa prompt
prompt = """Đánh giá bài viết IELTS sau theo 4 tiêu chí...
Prompt: {question}
Essay: {essay}
Evaluation:"""

# Thực hiện inference
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=1000)
evaluation = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### 4. Tạo dữ liệu tổng hợp

```bash
# Cấu hình API key cho Gemini
export GOOGLE_API_KEY="your-api-key"

# Chạy script tạo dữ liệu
cd data/
python synthetic_data.py
```

## Kết quả và hiệu suất

- **Dataset size:** 9,000+ mẫu thực tế từ IELTS 2022-2023
- **Model tốt nhất:** Mistral-7b sau DPO fine-tuning
- **Độ chính xác:** Cao trong việc đánh giá theo 4 tiêu chí IELTS
- **Khả năng:** Cung cấp feedback chi tiết và constructive cho người học

## Ứng dụng thực tế

1. **Cho giáo viên:** Hỗ trợ chấm bài và đưa ra feedback nhanh chóng
2. **Cho học sinh:** Tự đánh giá và cải thiện kỹ năng viết
3. **Cho trung tâm anh ngữ:** Tự động hóa quy trình đánh giá
4. **Cho nghiên cứu:** Cơ sở cho các nghiên cứu về NLP trong giáo dục

## Phát triển tương lai

- [ ] Tích hợp thêm các model SOTA khác
- [ ] Phát triển web application với giao diện thân thiện
- [ ] Mở rộng cho các task IELTS khác (Speaking, Reading)
- [ ] Cải thiện độ chính xác thông qua more human feedback
- [ ] Deploy model lên cloud để sử dụng rộng rãi

## Đóng góp (Contributing)

Mọi đóng góp cho dự án đều được hoan nghênh! Vui lòng:

1. Fork repository
2. Tạo branch mới cho feature
3. Commit changes
4. Push lên branch
5. Tạo Pull Request

## License

Dự án này được phát hành dưới MIT License. Xem file LICENSE để biết thêm chi tiết.

## Liên hệ

**Phạm Lê Ngọc Sơn**
- Email: phamlengocsononline@gmail.com
- GitHub: [GitHub Profile]
- LinkedIn: [LinkedIn Profile]

---

*Dự án này được phát triển bởi Phạm Lê Ngọc Sơn như một phần của nghiên cứu ứng dụng Machine Learning trong giáo dục. Đây là một công cụ hữu ích cho việc học và giảng dạy IELTS Writing Task 2.*

**⭐ Nếu dự án hữu ích, hãy star repo để ủng hộ tác giả! ⭐** 