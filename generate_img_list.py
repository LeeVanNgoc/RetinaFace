import os

# Thư mục chứa ảnh FDDB sau khi giải nén
base_dir = "data/FDDB/images"
output_file = "data/FDDB/img_list.txt"

# Duyệt qua tất cả các file trong thư mục ảnh và lưu đường dẫn tương đối (không có đuôi .jpg)
img_list = []
for root, _, files in os.walk(base_dir):
    for file in files:
        if file.lower().endswith('.jpg'):
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, base_dir)  # ví dụ: 2002/07/19/big/img_5.jpg
            img_id = os.path.splitext(rel_path)[0]           # loại bỏ .jpg
            img_list.append(img_id)

# Ghi ra file
with open(output_file, "w") as f:
    for img in sorted(img_list):
        f.write(img + "\n")

print(f"Đã tạo {len(img_list)} dòng trong {output_file}")
