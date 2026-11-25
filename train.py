import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Kendi model.py dosyamızı çağırdık
from model import RakamTanima

def egitim_ve_degerlendirme():
    #ekran kartı mı yoksa işlemci mi kullanılacağına karar verir (ekran kartı eğitimi daha hızlıdır)
    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(f"Kullanılan Donanım: {cihaz}")

    donusum = transforms.Compose([
        #yapay zeka büyük sayıları işlerken zorlanır. pikselleri 0 (siyah) ile 255 (beyaz) değerlerinden -0.4 ile +2.8 arasındaki sayılara çevirir
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    print("MNIST veri seti indiriliyor/yükleniyor...")
    #Eğitim için 60.000, Test için 10.000 el yazısı rakam görüntüsü
    egitim_verisi = datasets.MNIST(root='./data', train=True, download=True, transform=donusum)
    test_verisi = datasets.MNIST(root='./data', train=False, download=True, transform=donusum)

    #60.000 veride ram taşar. bu verileri 64lü paketler haline getirir.
    egitim_yukleyici = DataLoader(egitim_verisi, batch_size=64, shuffle=True)
    test_yukleyici = DataLoader(test_verisi, batch_size=1000, shuffle=False)

    #verinin hata oranını hesaplar ve buna göre bir puan verir. Sonraki adımda öğrenme hızı (lr) ile yavaş yavaş düzeltir
    model = RakamTanima().to(cihaz)
    kayip_fonksiyonu = nn.CrossEntropyLoss() 
    optimizasyon = optim.Adam(model.parameters(), lr=0.001)

    
    epoch_sayisi = 3 # test tekrar sayısı
    print(f"\nEğitim başlıyor ({epoch_sayisi} Epoch)...")
    baslangic_zamani = time.time()

    for epoch in range(1, epoch_sayisi + 1):
        model.train() # Modeli eğitim moduna al (Dropout aktif)
        toplam_kayip = 0

        for paket_no, (resimler, etiketler) in enumerate(egitim_yukleyici):
            resimler, etiketler = resimler.to(cihaz), etiketler.to(cihaz)

            optimizasyon.zero_grad()            # Önceki adımdan kalan hesaplamaları sıfırlar
            ciktilar = model(resimler)          # 64lü paketleri modele verir ve model tahminlerde bulunur
            kayip = kayip_fonksiyonu(ciktilar, etiketler) # tahminleri kontrol eder ve hata puanını hesaplar
            kayip.backward()                    # Geri yayılım (Backpropagation): hatanın hangi nöronun ne kadar suçlu olduğunu bulur
            optimizasyon.step()                 # suçlu nöronları düzeltir

            toplam_kayip += kayip.item()

            if paket_no % 300 == 0:
                print(f"Epoch {epoch}/{epoch_sayisi} | Adım: {paket_no}/{len(egitim_yukleyici)} | Anlık Kayıp (Loss): {kayip.item():.4f}")

    egitim_suresi = time.time() - baslangic_zamani
    print(f"\n---> Eğitim Tamamlandı! Toplam Eğitim Süresi: {egitim_suresi:.2f} saniye <---")

    
    model.eval() # eğitimin bittiği ve test aşamasının başladığı yer
    test_kaybi = 0
    dogru_tahmin = 0
    hatali_ornekler = [] # Yanlış tahmin edilen resimleri raporlamak için liste

    with torch.no_grad(): # Test sırasında türev hesaplamayı kapat (Hız ve bellek tasarrufu)
        for resimler, etiketler in test_yukleyici:
            resimler, etiketler = resimler.to(cihaz), etiketler.to(cihaz)
            ciktilar = model(resimler)
            test_kaybi += kayip_fonksiyonu(ciktilar, etiketler).item()
            
            
            tahminler = ciktilar.argmax(dim=1, keepdim=True) #modelin tahmininin son kararı
            dogru_mu = tahminler.eq(etiketler.view_as(tahminler)) #test tahminleri etiketlere eşit mi
            dogru_tahmin += dogru_mu.sum().item()

            #yanlış yapılan ilk 10 hatayı gerçek cevap ile birlikte listeler 
            if len(hatali_ornekler) < 10:
                yanlis_indeksler = (dogru_mu == False).nonzero(as_tuple=True)[0]
                for idx in yanlis_indeksler:
                    if len(hatali_ornekler) < 10:
                        hatali_ornekler.append((
                            resimler[idx].cpu(),
                            etiketler[idx].cpu().item(),
                            tahminler[idx].cpu().item()
                        ))

    toplam_test_verisi = len(test_yukleyici.dataset)
    dogruluk_orani = 100. * dogru_tahmin / toplam_test_verisi

    print("\n================ SONUÇ RAPORU ================")
    print(f"Toplam Eğitim Süresi : {egitim_suresi:.2f} saniye")
    print(f"Test Veri Sayısı     : {toplam_test_verisi}")
    print(f"Doğru Tahmin Sayısı  : {dogru_tahmin}")
    print(f"Hatalı Tahmin Sayısı : {toplam_test_verisi - dogru_tahmin}")
    print(f"Test Doğruluğu       : %{dogruluk_orani:.2f}")
    print("==============================================")

    # burada save alınır
    model_dosya_adi = "mnist_cnn.pth"
    torch.save(model.state_dict(), model_dosya_adi) #bilgisayar kapandığında modelin eğitimi silinmemesi için kullanılır
    print(f"\n[BİLGİ] Eğitilen model '{model_dosya_adi}' olarak başarıyla kaydedildi.")

    #istenilen hatalı tahmibler raporunu veri
    if hatali_ornekler:
        fig, eksenler = plt.subplots(2, 5, figsize=(12, 5))
        fig.suptitle(f"Modelin Yanlış Tahmin Ettiği Örnekler (Test Doğruluğu: %{dogruluk_orani:.2f})", fontsize=14)
        for i, eksen in enumerate(eksenler.flat):
            if i < len(hatali_ornekler):
                resim, gercek, tahmin = hatali_ornekler[i]
                eksen.imshow(resim.squeeze(), cmap='gray')
                eksen.set_title(f"Gerçek: {gercek} | Tahmin: {tahmin}", color="red")
                eksen.axis('off')
        plt.tight_layout()
        plt.savefig("hatali_tahminler_raporu.png")
        print("[BİLGİ] Hatalı tahminlerin görsel raporu 'hatali_tahminler_raporu.png' olarak kaydedildi.")

if __name__ == "__main__":
    egitim_ve_degerlendirme()