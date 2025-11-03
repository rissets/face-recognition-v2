import cv2
from deepface import DeepFace
import time
import logging
import numpy as np

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('antispoof_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

print("Membuka webcam...")
logger.info("Memulai aplikasi anti-spoofing")

# Coba ganti '0' dengan '1' jika Anda punya lebih dari satu webcam
cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    logger.error("Error: Tidak dapat membuka webcam.")
    exit()

# Set resolusi webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Variabel untuk menghitung FPS (Frames Per Second)
prev_frame_time = 0
frame_count = 0
detection_count = 0

print("\nMemulai stream...")
print("-> Tekan 'q' di jendela webcam untuk keluar.")
print("-> Saat pertama kali dijalankan, model akan diunduh (~70MB). Mohon tunggu.\n")
logger.info("Stream dimulai")

while True:
    # Baca frame demi frame
    ret, frame = cap.read()
    if not ret:
        logger.error("Error: Gagal membaca frame dari webcam.")
        break

    # Balik frame secara horizontal (efek cermin) agar lebih intuitif
    frame = cv2.flip(frame, 1)
    frame_count += 1

    # Status info di frame
    status_text = "Mencari wajah..."
    status_color = (255, 255, 0)  # Kuning

    try:
        logger.debug(f"Frame {frame_count}: Memproses deteksi anti-spoofing")
        
        # Panggil DeepFace.analyze()
        # Kita hanya butuh aksi 'spoof' (anti-spoofing)
        # enforce_detection=False agar tidak error jika tidak ada wajah di frame
        results = DeepFace.analyze(
            img_path = frame, 
            actions = ['spoof'], 
            enforce_detection = False,
            silent = True  # Mengurangi output verbose
        )
        
        logger.debug(f"Frame {frame_count}: Hasil analisis - {len(results)} wajah terdeteksi")
        
        # 'results' adalah list, berisi satu dictionary untuk setiap wajah
        if results and len(results) > 0:
            detection_count += 1
            logger.info(f"Frame {frame_count}: Wajah terdeteksi - Total deteksi: {detection_count}")
            
            for i, face_result in enumerate(results):
                # Ambil koordinat kotak wajah
                region = face_result['region']
                x = region['x']
                y = region['y']
                w = region['w']
                h = region['h']
                
                # Ambil hasil anti-spoofing
                spoof_data = face_result['spoof']
                is_real = spoof_data['is_real']
                confidence = spoof_data['confidence']
                
                # Log hasil deteksi
                result_status = "ASLI" if is_real else "PALSU (SPOOF)"
                logger.info(f"Frame {frame_count}, Wajah {i+1}: {result_status} - Confidence: {confidence:.1%}")
                
                # Tentukan label dan warna berdasarkan hasilnya
                if is_real:
                    label = f"ASLI ({confidence:.1%})"
                    color = (0, 255, 0) # Hijau
                    status_text = f"Wajah ASLI terdeteksi!"
                    status_color = (0, 255, 0)
                else:
                    label = f"PALSU (SPOOF) ({confidence:.1%})"
                    color = (0, 0, 255) # Merah
                    status_text = f"WAJAH PALSU terdeteksi!"
                    status_color = (0, 0, 255)
                
                # Gambar kotak di sekitar wajah
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                
                # Tulis label di atas kotak
                cv2.putText(frame, label, (x, y - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Tambahkan info koordinat
                cv2.putText(frame, f"#{i+1}", (x+5, y+20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            logger.debug(f"Frame {frame_count}: Tidak ada wajah terdeteksi")

    except Exception as e:
        # Tangani jika ada error selama analisis
        logger.error(f"Frame {frame_count}: Error saat analisis - {str(e)}")
        status_text = f"Error: {str(e)}"
        status_color = (0, 0, 255)

    # Hitung dan tampilkan FPS
    new_frame_time = time.time()
    if prev_frame_time != 0:
        fps = 1 / (new_frame_time - prev_frame_time)
    else:
        fps = 0
    prev_frame_time = new_frame_time
    
    # Tampilkan informasi di frame
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    cv2.putText(frame, f"Frame: {frame_count}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.putText(frame, f"Deteksi: {detection_count}", (10, 85), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.putText(frame, status_text, (10, frame.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # Tampilkan frame yang sudah diproses
    cv2.imshow("Deteksi Anti-Spoofing DeepFace (Tekan 'q' untuk keluar)", frame)

    # Cek jika tombol 'q' ditekan untuk keluar
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        logger.info("Aplikasi dihentikan oleh user")
        break
    elif key == ord('s'):
        # Save screenshot
        screenshot_name = f"screenshot_{int(time.time())}.jpg"
        cv2.imwrite(screenshot_name, frame)
        logger.info(f"Screenshot disimpan: {screenshot_name}")

# Bersihkan setelah loop selesai
print("\nMenutup stream...")
logger.info(f"Statistik akhir - Total frame: {frame_count}, Total deteksi: {detection_count}")
cap.release()
cv2.destroyAllWindows()