import os
import torch
from torchvision import transforms, datasets
from PIL import Image, UnidentifiedImageError

# Kendi model mizı çağırıyoruz
from model import RakamTanima

# 1x1x28x28 boyutundaki tensörleri kullanmak zorunayız. Bu nedenle resmi 28x28 boyutuna getirip normalize ediyoruz.
def goruntu_on_isleme(goruntu_yolu): 
    
    # uygun dosya kontrolü yapılır
    if not os.path.exists(goruntu_yolu):
        raise FileNotFoundError(f"Görüntü dosyası bulunamadı: '{goruntu_yolu}'")

    # Görüntüyü açarken oluşabilecek hataları yakala ve kullanıcıya anlamlı bir mesaj ver
    try:
        img = Image.open(goruntu_yolu)
    except (UnidentifiedImageError, Exception) as e:
        raise ValueError(f"Geçersiz dosya biçimi! '{goruntu_yolu}' okunabilir bir resim dosyası değil.") from e

    # Boyut ve Biçim Dönüştürme:
    # 'L' modu ile resmi 1 kanallı Gri Tonlamaya (Grayscale) çevirir
    # 28x28 piksel boyutuna getirir
    # Tensöre çevirip eğitimdeki aynı değerlerle normalize eder 
    img = img.convert('L')
    donusum = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Modele girmesi için başına (batch size) boyutu ekle: (1, 28, 28) --> (1, 1, 28, 28)
    tensor_img = donusum(img).unsqueeze(0)
    return tensor_img

# Tahmin fonksiyonu: Görüntü yolunu alır, modeli yükler, görüntüyü işler ve tahmin yapar
def rakam_tahmin_et(goruntu_yolu, model_yolu="mnist_cnn.pth"):
    print(f"\n--- Tahmin İşlemi Başlatılıyor: '{goruntu_yolu}' (Model: '{model_yolu}') ---")

    # 1. model dosyasının varlığını kontrol eder. eğer yoksa kullanıcıya yeni bir dosya oluşturması gerektiğini söyler.
    if not os.path.exists(model_yolu):
        print(f"[HATA - Eksik Model] '{model_yolu}' dosyası bulunamadı! Lütfen önce train.py ile modeli eğitin.")
        return None

    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    try:
        model = RakamTanima().to(cihaz) # boş beyin iskeleti kurulur
        # boş beynin içine eğitilmiş ağırlıklar yüklenir. Eğer model dosyası bozuksa hata verir.
        model.load_state_dict(torch.load(model_yolu, map_location=cihaz, weights_only=True)) 
        model.eval()
    except Exception as e:
        print(f"[HATA - Bozuk Model] Model dosyası yüklenirken hata oluştu: {e}")
        return None

    # model geçersizse bu blok çalışır 
    try:
        girdi_tensoru = goruntu_on_isleme(goruntu_yolu).to(cihaz)
    except (FileNotFoundError, ValueError) as hata_mesaji:
        print(f"[HATA - Geçersiz Girdi] {hata_mesaji}")
        return None

    
    with torch.no_grad():
        # resme bakar ve 10 rakam için puan verir 
        ciktilar = model(girdi_tensoru) 
        olasiliklar = torch.softmax(ciktilar, dim=1) # puanları % olasılığa çevir
        guven_orani, tahmin = torch.max(olasiliklar, dim=1)  # en yüksek olasılığı ve tahmin edilen rakamı alır

    tahmin_edilen_rakam = tahmin.item()
    guven_yuzdesi = guven_orani.item() * 100

    print(f"[BAŞARILI] Tahmin Edilen Rakam: {tahmin_edilen_rakam} (Güven Oranı: %{guven_yuzdesi:.2f})")
    return tahmin_edilen_rakam

def test_ortami_hazirla():
    # test için örnek bir resim oluşturur ve .png formatında kaydeder
    test_seti = datasets.MNIST(root='./data', train=False, download=True)
    ornek_resim, gercek_etiket = test_seti[0] 
    ornek_resim.save("ornek_rakam.png")
    print(f"[HAZIRLIK] Test için 'ornek_rakam.png' oluşturuldu (Gerçek değeri: {gercek_etiket}).") 

    # Geçersiz dosya testi için içine metin yazılmış sahte bir .txt dosyası oluşturur
    with open("gecersiz_dosya.txt", "w") as f:
        f.write("Bu bir resim dosyasi degildir, hata testi icindir.")
    print("[HAZIRLIK] Hata testi için 'gecersiz_dosya.txt' oluşturuldu.")

if __name__ == "__main__":
    print("================ UYGULAMA VE HATA TESTLERİ ================")
    test_ortami_hazirla()

    # TEST 1: Başarılı Tahmin Testi (Geçerli Görüntü + Geçerli Model)
    print("\n>>> TEST 1: Geçerli Görüntü Tahmini")
    rakam_tahmin_et("ornek_rakam.png", model_yolu="mnist_cnn.pth")

    # TEST 2: Geçersiz Dosya Biçimi Testi (Resim olmayan .txt dosyası verilmesi)
    print("\n>>> TEST 2: Geçersiz Dosya Biçimi Testi")
    rakam_tahmin_et("gecersiz_dosya.txt", model_yolu="mnist_cnn.pth")

    # TEST 3: Olmayan Görüntü Dosyası Testi
    print("\n>>> TEST 3: Bulunamayan Görüntü Dosyası Testi")
    rakam_tahmin_et("olmayan_resim.png", model_yolu="mnist_cnn.pth")

    # TEST 4: Eksik Model Dosyası Testi (Yanlış/olmayan model yolu verilmesi)
    print("\n>>> TEST 4: Eksik Model Dosyası Testi")
    rakam_tahmin_et("ornek_rakam.png", model_yolu="olmayan_model.pth")

    #Bizim veri verip modelin tahmin ettiği manuel sistem
    print("\n>>> MANUEL TEST: Kullanıcıdan Görüntü Yolu Alarak Tahmin")
    manuel_girdi = "manuel_rakam.png"  # Burada kullanıcı kendi resim yolunu girebilir
    rakam_tahmin_et(manuel_girdi, model_yolu="mnist_cnn.pth")
    print("\n===========================================================")