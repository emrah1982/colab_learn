#!/usr/bin/env python3
# main.py - YOLO11 Training Main Script for Colab

import os
import sys
from pathlib import Path
import shutil
from datetime import datetime

# Import components
from setup_utils import check_gpu, install_required_packages
from hyperparameters import create_hyperparameters_file, load_hyperparameters
from dataset_utils import download_dataset, fix_directory_structure, update_dataset_yaml
from memory_utils import show_memory_usage, clean_memory
from training import train_model, save_to_drive
from model_downloader import download_yolo11_models, download_specific_model_type

# Check if running in Colab
def is_colab():
    """Check if running in Google Colab"""
    try:
        import google.colab
        print("Google Colab environment detected.")
        return True
    except:
        print("Running in local environment.")
        return False

def mount_google_drive():
    """Google Drive'ı bağla"""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("Google Drive başarıyla bağlandı.")
        return True
    except Exception as e:
        print(f"Google Drive bağlanırken hata oluştu: {e}")
        return False

def save_models_to_drive(drive_folder_path, best_file=True, last_file=True):
    """En iyi ve son model dosyalarını Google Drive'a kaydet"""
    if not is_colab():
        print("Bu fonksiyon sadece Google Colab'da çalışır.")
        return False
    
    # Google Drive'ın bağlı olduğunu kontrol et
    if not os.path.exists('/content/drive'):
        if not mount_google_drive():
            return False
    
    # Kaynak klasörü kontrol et
    source_dir = "runs/train/exp/weights"
    if not os.path.exists(source_dir):
        print(f"Kaynak klasör bulunamadı: {source_dir}")
        return False
    
    # Hedef klasörü oluştur
    os.makedirs(drive_folder_path, exist_ok=True)
    
    # Dosyaları kopyala
    copied_files = []
    
    if best_file and os.path.exists(os.path.join(source_dir, "best.pt")):
        shutil.copy2(os.path.join(source_dir, "best.pt"), os.path.join(drive_folder_path, "best.pt"))
        copied_files.append("best.pt")
    
    if last_file and os.path.exists(os.path.join(source_dir, "last.pt")):
        shutil.copy2(os.path.join(source_dir, "last.pt"), os.path.join(drive_folder_path, "last.pt"))
        copied_files.append("last.pt")
    
    if copied_files:
        print(f"Google Drive'a kaydedilen dosyalar: {', '.join(copied_files)}")
        print(f"Kaydedilen konum: {drive_folder_path}")
        return True
    else:
        print("Kopyalanacak dosya bulunamadı.")
        return False

def download_models_menu():
    """Interactive menu for downloading YOLO11 models"""
    print("\n===== YOLO11 Model Download =====")
    
    # Ask for save directory - Updated default path here
    default_dir = os.path.join("/content/colab_learn", "yolo11_models")
    save_dir = input(f"\nSave directory (default: {default_dir}): ") or default_dir
    
    # Ask for download type
    print("\nDownload options:")
    print("1. Download single model")
    print("2. Download all detection models")
    print("3. Download all models (all types)")
    
    choice = input("\nYour choice (1-3): ")
    
    if choice == "1":
        # Single model download
        print("\nSelect model type:")
        print("1. Detection (default)")
        print("2. Segmentation")
        print("3. Classification")
        print("4. Pose")
        print("5. OBB (Oriented Bounding Box)")
        
        model_type_map = {
            "1": "detection",
            "2": "segmentation",
            "3": "classification",
            "4": "pose",
            "5": "obb"
        }
        
        model_type_choice = input("Enter choice (1-5, default: 1): ") or "1"
        model_type = model_type_map.get(model_type_choice, "detection")
        
        # Ask for model size
        print("\nSelect model size:")
        print("1. Small (s)")
        print("2. Medium (m) (default)")
        print("3. Large (l)")
        print("4. Extra Large (x)")
        
        size_map = {
            "1": "s",
            "2": "m",
            "3": "l",
            "4": "x"
        }
        
        size_choice = input("Enter choice (1-4, default: 2): ") or "2"
        size = size_map.get(size_choice, "m")
        
        # Download the selected model
        model_path = download_specific_model_type(model_type, size, save_dir)
        if model_path:
            print(f"\nModel downloaded successfully to: {model_path}")
    
    elif choice == "2":
        # Download all detection models
        detection_models = ["yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]
        downloaded = download_yolo11_models(save_dir, detection_models)
        print(f"\nDownloaded {len(downloaded)} detection models to {save_dir}")
    
    elif choice == "3":
        # Download all models
        downloaded = download_yolo11_models(save_dir)
        print(f"\nDownloaded {len(downloaded)} models to {save_dir}")
    
    else:
        print("\nInvalid choice. No models downloaded.")
        return None
    
    return save_dir

def interactive_setup():
    """Interactively collect training parameters from the user"""
    print("\n===== YOLO11 Training Setup =====")

    # İki farklı Roboflow URL'si sorma
    roboflow_urls = []
    roboflow_url1 = input("\nİlk Roboflow URL (https://universe.roboflow.com/ds/...): ").strip()
    if roboflow_url1:
        roboflow_urls.append(roboflow_url1)
        
        add_second = input("\nİkinci bir Roboflow URL eklemek istiyor musunuz? (e/h, varsayılan: h): ").lower() or "h"
        if add_second.startswith("e"):
            roboflow_url2 = input("İkinci Roboflow URL: ").strip()
            if roboflow_url2:
                roboflow_urls.append(roboflow_url2)

    # Proje kategorisini sor (hastalık, böcek vb.)
    print("\nProje kategorisini seçin:")
    print("1) Hastalık (disease)")
    print("2) Böcek (insect)")
    print("3) Diğer (özel belirtin)")
    
    while True:
        category_choice = input("\nSeçiminiz [1-3] (varsayılan: 1): ") or "1"
        category_options = {
            "1": "diseases",
            "2": "insect",
            "3": "other"
        }
        
        if category_choice in category_options:
            category = category_options[category_choice]
            if category == "other":
                category = input("Özel kategori adını girin: ")
            break
        print("Lütfen 1-3 arasında bir sayı seçin.")

    # Kaldığı yerden devam etme seçeneği
    resume_training = False
    resume_from_drive = False
    checkpoint_path = None
    
    if is_colab():
        has_previous = input("\nEğitimi kaldığı yerden devam ettirmek istiyor musunuz? (e/h, varsayılan: h): ").lower() or "h"
        
        if has_previous.startswith("e"):
            resume_training = True
            resume_from_drive = input("Önceki modeli Google Drive'dan yüklemek istiyor musunuz? (e/h, varsayılan: e): ").lower() or "e"
            
            if resume_from_drive.startswith("e"):
                # Google Drive'ı bağla
                if not os.path.exists('/content/drive'):
                    mount_google_drive()
                
                # Drive'daki model yolunu sor
                base_folder = f"/content/drive/MyDrive/Tarim/Kodlar/colab_egitim/{category}"
                print(f"\nTahmini model klasörü: {base_folder}")
                
                custom_path = input(f"Model yolunu onaylayın veya yeni bir yol girin (varsayılan: {base_folder}): ") or base_folder
                
                # last.pt veya best.pt dosyasını seç
                model_type = input("\nHangi model dosyasını kullanmak istiyorsunuz? (best/last, varsayılan: best): ").lower() or "best"
                if model_type not in ["best", "last"]:
                    model_type = "best"
                
                checkpoint_path = os.path.join(custom_path, f"{model_type}.pt")
                
                if os.path.exists(checkpoint_path):
                    print(f"Model dosyası bulundu: {checkpoint_path}")
                    
                    # Hedef klasöre kopyala
                    os.makedirs("runs/train/exp/weights", exist_ok=True)
                    shutil.copy2(checkpoint_path, f"runs/train/exp/weights/{model_type}.pt")
                    print(f"Model dosyası eğitim klasörüne kopyalandı.")
                else:
                    print(f"UYARI: Model dosyası bulunamadı: {checkpoint_path}")
                    print("Eğitime sıfırdan başlanacak.")
                    resume_training = False

    # Ask for number of epochs
    while True:
        try:
            epochs = int(input("\nEpoch sayısı [100-5000 önerilen] (varsayılan: 500): ") or "500")
            if epochs <= 0:
                print("Lütfen pozitif bir sayı girin.")
                continue
            break
        except ValueError:
            print("Lütfen geçerli bir sayı girin.")

    # Ask for model size
    print("\nModel boyutunu seçin:")
    print("1) yolo11n.pt - Nano (en hızlı, en düşük doğruluk)")
    print("2) yolo11s.pt - Small (hızlı, orta doğruluk)")
    print("3) yolo11m.pt - Medium (dengeli)")
    print("4) yolo11l.pt - Large (yüksek doğruluk, yavaş)")
    print("5) yolo11x.pt - XLarge (en yüksek doğruluk, en yavaş)")

    while True:
        model_choice = input("\nSeçiminiz [1-5] (varsayılan: 3): ") or "3"
        model_options = {
            "1": "yolo11n.pt",
            "2": "yolo11s.pt",
            "3": "yolo11m.pt",
            "4": "yolo11l.pt",
            "5": "yolo11x.pt"
        }
        if model_choice in model_options:
            model = model_options[model_choice]
            
            # Check if model exists, offer to download if not
            # Updated default model directory path
            model_dir = os.path.join("/content/colab_learn", "yolo11_models")
            model_path = os.path.join(model_dir, model)
            
            if not os.path.exists(model_path):
                print(f"\nModel {model} yerel olarak bulunamadı.")
                download_now = input("Şimdi indirmek ister misiniz? (e/h, varsayılan: e): ").lower() or "e"
                
                if download_now.startswith("e"):
                    os.makedirs(model_dir, exist_ok=True)
                    download_specific_model_type("detection", model[6], model_dir)
                else:
                    print(f"Model eğitim sırasında otomatik olarak indirilecek.")
            
            break
        print("Lütfen 1-5 arasında bir sayı seçin.")

    # Ask for batch size
    while True:
        try:
            batch_size = int(input("\nBatch size (varsayılan: 16, düşük RAM için 8 veya 4 önerilir): ") or "16")
            if batch_size <= 0:
                print("Lütfen pozitif bir sayı girin.")
                continue
            break
        except ValueError:
            print("Lütfen geçerli bir sayı girin.")

    # Ask for image size
    while True:
        try:
            img_size = int(input("\nGörüntü boyutu (varsayılan: 640, 32'nin katı olmalı): ") or "640")
            if img_size <= 0 or img_size % 32 != 0:
                print("Lütfen 32'nin katı olan pozitif bir sayı girin.")
                continue
            break
        except ValueError:
            print("Lütfen geçerli bir sayı girin.")

    # Drive'a kayıt klasörü
    drive_save_path = None
    if is_colab():
        print("\nEğitim sonuçlarını Google Drive'a kaydetme ayarları:")
        save_to_drive_opt = input("Eğitim sonuçlarını Google Drive'a kaydetmek istiyor musunuz? (e/h, varsayılan: e): ").lower() or "e"
        
        if save_to_drive_opt.startswith("e"):
            # Google Drive'ı bağla
            if not os.path.exists('/content/drive'):
                mount_google_drive()
            
            # Drive'daki model yolunu sor
            default_drive_path = f"/content/drive/MyDrive/Tarim/Kodlar/colab_egitim/{category}"
            drive_save_path = input(f"Modellerin kaydedileceği klasörü belirtin (varsayılan: {default_drive_path}): ") or default_drive_path
            
            # Otomatik olarak tarih/saat ekleme
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            drive_save_path = os.path.join(drive_save_path, timestamp)
            print(f"Modeller şu klasöre kaydedilecek: {drive_save_path}")
    
    # Option to use hyperparameter file
    use_hyp = input("\nHiperparametre dosyası kullanılsın mı (hyp.yaml)? (e/h) (varsayılan: e): ").lower() or "e"
    use_hyp = use_hyp.startswith("e")

    # Automatically check GPU/CPU
    device = check_gpu()

    # Return parameters as options dictionary
    options = {
        'roboflow_urls': roboflow_urls,
        'epochs': epochs,
        'model': model,
        'batch': batch_size,
        'imgsz': img_size,
        'device': device,  # Automatically determined from GPU check
        'data': 'dataset.yaml',
        'project': 'runs/train',
        'name': 'exp',
        'pretrained': True,
        'optimizer': 'auto',
        'verbose': True,
        'exist_ok': True,
        'resume': resume_training,
        'use_hyp': use_hyp,
        'category': category,
        'drive_save_path': drive_save_path,
        'checkpoint_path': checkpoint_path
    }

    print("\n===== Seçilen Parametreler =====")
    for key, value in options.items():
        print(f"{key}: {value}")

    confirm = input("\nBu parametrelerle devam etmek istiyor musunuz? (e/h): ").lower()
    if confirm != 'e' and confirm != 'evet':
        print("Kurulum iptal edildi.")
        return None

    return options

def main():
    """Main function - optimized for Colab"""
    print("\n===== YOLO11 Training Framework =====")
    print("1. Model indirme")
    print("2. Eğitim kurulumu")
    print("3. Çıkış")
    
    choice = input("\nBir seçenek seçin (1-3): ")
    
    if choice == "1":
        # Download models
        download_models_menu()
        # After downloading, ask if they want to continue to training
        train_now = input("\nEğitim kurulumuna geçmek ister misiniz? (e/h, varsayılan: e): ").lower() or "e"
        if not train_now.startswith("e"):
            return
        
    if choice == "1" or choice == "2":
        # Check environment
        in_colab = is_colab()
        
        # Install required packages from requirements.txt
        install_required_packages()

        # Create or use existing hyperparameter file
        hyp_path = create_hyperparameters_file()
        hyperparameters = load_hyperparameters(hyp_path)

        # Interactive setup
        options = interactive_setup()
        if options is None:
            return

        # Download dataset if Roboflow URL is provided
        if 'roboflow_url' in options and options['roboflow_url']:
            if not download_dataset(options['roboflow_url']):
                print('Veri seti indirilemedi. Çıkılıyor...')
                return
        else:
            print('Lütfen geçerli bir Roboflow URL girin.')
            return

        # Train the model
        results = train_model(options, hyp=hyperparameters, resume=options.get('resume', False), epochs=options['epochs'])

        if results:
            print('Eğitim tamamlandı!')
            print(f'Sonuçlar: {results}')
            
            # Google Drive'a kaydet
            if in_colab and options.get('drive_save_path'):
                print("\nModeller Google Drive'a kaydediliyor...")
                if save_models_to_drive(options['drive_save_path']):
                    print("Modeller Google Drive'a başarıyla kaydedildi.")
                else:
                    print("Google Drive'a kaydetme işlemi başarısız oldu.")
        else:
            print('Eğitim başarısız oldu veya yarıda kesildi.')
            
            # Yarım kalan eğitimi Drive'a kaydet
            if in_colab and options.get('drive_save_path'):
                save_anyway = input("\nYarım kalan eğitimi Google Drive'a kaydetmek istiyor musunuz? (e/h, varsayılan: e): ").lower() or "e"
                if save_anyway.startswith("e"):
                    print("\nYarım kalan model Google Drive'a kaydediliyor...")
                    if save_models_to_drive(options['drive_save_path']):
                        print("Model Google Drive'a başarıyla kaydedildi.")
                    else:
                        print("Google Drive'a kaydetme işlemi başarısız oldu.")
    
    elif choice == "3":
        print("Çıkılıyor...")
    
    else:
        print("Geçersiz seçenek. Çıkılıyor...")

if __name__ == "__main__":
    try:
        # Call main function
        main()
    except KeyboardInterrupt:
        print("\nKullanıcı tarafından durduruldu. Çıkılıyor...")
    except Exception as e:
        print(f"\nHata oluştu: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nİşlem tamamlandı.")
