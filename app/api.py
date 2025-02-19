import numpy as np  # Import numpy để chuyển đổi ndarray
from fastapi import APIRouter, File, Form, UploadFile
import os
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from app.inference import *
import pretty_midi
from datetime import datetime
from fastapi import HTTPException
from typing import Dict

router = APIRouter()

@router.post("/nhan-gen-melody-with-midi-multitone")
async def nhan_create_melody(username: str = Form(...), file: UploadFile = File(...)):
    try:
        # Tạo thư mục nếu chưa tồn tại
        upload_dir = "uploads"
        results_dir = "results"
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        # Lưu file MIDI đã tải lên
        file_upload = os.path.join(upload_dir, file.filename)
        with open(file_upload, "wb") as f:
            f.write(await file.read())

        # Gọi hàm generate_melody để tạo nhạc
        Music, Melody_midi = nhan_generate_melody(file_upload, Note_Count=100)

        # Tìm index lớn nhất của các file có cùng cú pháp tên
        max_index = 0
        for file_name in os.listdir(results_dir):
            if file_name.startswith(f"{username}_yes_midi_multitone_") and file_name.endswith(".midi"):
                try:
                    index_part = file_name.split("_yes_midi_multitone_")[1].split(".midi")[0]
                    index = int(index_part)
                    max_index = max(max_index, index)
                except (IndexError, ValueError):
                    pass  # Bỏ qua các file không hợp lệ

        # Tạo file mới với index tăng lên
        new_index = max_index + 1
        file_name = f"{username}_yes_midi_multitone_{new_index}.midi"
        file_path = os.path.join(results_dir, file_name)

        # Lưu file MIDI đã tạo
        Melody_midi.write('midi', fp=file_path)

        # Ghép file MIDI đầu vào và file MIDI đã tạo từ Melody_midi
        input_midi = pretty_midi.PrettyMIDI(file_upload)
        generated_midi = pretty_midi.PrettyMIDI(file_path)
        for instrument in generated_midi.instruments:
            input_midi.instruments.append(instrument)

        # Lưu file MIDI hợp nhất
        final_output_path = os.path.join(results_dir, file_name)
        input_midi.write(final_output_path)

        # Cập nhật đường dẫn vào database
        full_url = f"{file_name}"
        update_record("musics", "title", file_name.replace(".midi", ""), "title = 'Clone title'")
        update_record("musics", "full_url", full_url, "full_url = 'http://example.com/xxx'")

        # Trả về file kết quả
        return FileResponse(path=final_output_path, filename=file_name, media_type="audio/midi")

    except Exception as e:
        return {"error": str(e)}

@router.post("/hoang-gen-melody-without-midi-multitone")
async def hoang_create_melody(username: str = Form(...)):
    prediction_output = generate()
    file_path, file_name = hoang_create_midi(prediction_output, username)

    # Trả về thông tin file và file để download
    return JSONResponse(content={
        "filename": file_name,
        "file_url": f"/{file_path}"
    })

@router.post("/roll-gen-melody-without-midi-monotone")
async def roll_create_melody1(username: str = Form(...)):
    # Load model và mapping
    model, mappings = roll_load_model_and_mapping()
    seed = "67 _ 67 _ 67 _ _ 65 64 _ 64 _ 64 _ _"
    # Generate melody
    melody = roll_generate_melody(model, mappings, seed, num_steps=1500, max_sequence_length=5000, temperature=0.3, sequence_length=64)
    print(melody)
    # Save melody
    file_name, file_path = roll_save_melody_without_midi(melody, username)

    # Trả về thông tin file và file để download
    return JSONResponse(content={
        "filename": file_name,
        "file_url": f"/{file_path}"
    })


@router.post("/roll-gen-melody-with-midi-monotone")
async def roll_create_melody2(username: str = Form(...), file: UploadFile = File(...)):
    try:
        # Tạo thư mục lưu trữ tệp tải lên
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)

        # Lưu tệp MIDI đã tải lên
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Đọc và mã hóa tệp MIDI
        midi_file = m21.converter.parse(file_path)
        encoded_seed = roll_encode_song(midi_file)
        
        print(encoded_seed)

        # Load model và mapping
        model, mappings = roll_load_model_and_mapping()

        try:
            melody = roll_generate_melody(model, mappings, encoded_seed, num_steps=1500, max_sequence_length=5000, temperature=0.3, sequence_length=64)
        except KeyError as e:
            return {"error": f"Token {str(e)} is not in the mappings. Please check your seed or mappings."}

        
        file_name, file_path = roll_save_melody_with_midi(melody,username)

        # Trả về file MIDI dưới dạng FileResponse
        return FileResponse(path=file_path, filename=file_name, media_type="audio/midi")

    except Exception as e:
        return {"error": str(e)}

@router.get("/results/{filename}")
async def download_page(filename: str):
    # Nếu filename không có phần mở rộng, mặc định thêm .midi
    if not filename.endswith(".midi"):
        filename += ".midi"
    file_path = os.path.join('results', filename)
    # Kiểm tra file tồn tại
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    # Trả về file để tải
    return FileResponse(file_path, media_type='audio/midi', filename=filename)

@router.put("/api/music/{music_id}/update-title")
async def update_music_title(music_id: int, request_body: Dict[str, str]):
    new_title = request_body.get("new_title")
    new_url = new_title
    old_title = get_music_title_by_id(music_id)    
    rename_file_in_results(old_title["title"],new_title)   
    update_music_title_by_id(music_id, new_title)  # Cập nhật tiêu đề bài hát
    update_music_url_by_id(music_id, new_url)
    return {"message": "Music title updated successfully"}

@router.post("/api/update-purchase-status")
async def update_purchase_status(musicId: str):
    print(musicId)
    update_music_isPurchased_by_id(int(musicId))
    if not musicId:
        raise HTTPException(status_code=400, detail="Music ID không hợp lệ.")
    return {"message": "Trạng thái mua hàng đã được cập nhật!", "musicId": musicId}