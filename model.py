import torch #hızlı matematiksel işlem yapmamızı sağlar
import torch.nn as nn #yapay sinir ağı katmanları için (nn = neural network)
import torch.nn.functional as F #matematiksel fonksiyonlar içerir

class RakamTanima(nn.Module): #kalıtım alıyoruz, nn.Module sınıfından türetiyoruz
    def __init__ (self): #yapıcı metod
        super(RakamTanima, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1,) #28x28 olan veriyi kenarlarından kırpıldığı için 26x26 boyutuna düşürür
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1,) #32 yi 64 e çıkarır, 24x24 boyutuna düşürür
        
        self.dropout1 = nn.Dropout(0.25) #eğitim sırasında nöronların %25 ini rastgele kapatarak overfittingi önler
        self.dropout2 = nn.Dropout(0.5) #eğitim sırasında nöronların %50 sini rastgele kapatarak overfittingi önler

        self.fc1 = nn.Linear(64 * 12 * 12, 128) #9216 boyutundaki veriyi 128 boyutuna düşürür
        self.fc2 = nn.Linear(128, 10) #128 boyutundaki veriyi 10 boyutuna düşürür, 0 dan 9 a kadar olan rakamları temsil eder

    def forward (self, x): #bu metod, yeni resim geldiğinde bu resmin hangi işlemlerden geçeceğini belirler
        x = self.conv1(X) #(1, 28, 28) boyutundaki veriyi (32, 26, 26) boyutuna düşürür
        x = F.relu(x) #negatif değerleri 0 yapar, pozitif değerleri ise değiştirmez

        X = self.conv2(x) #(32, 26, 26) boyutundaki veriyi (64, 24, 24) boyutuna düşürür
        x = F.relu(x) 

        x = F.max_pool2d(x, 2) #(64, 24, 24) boyutundaki veriyi (64, 12, 12) boyutuna düşürür 2x2 lik 4 pikselden en büyük olanı alır, resmi yarıya indirir
        X = self.dropout1(x) #eğitim sırasında nöronların %25 ini rastgele kapatarak overfittingi önler

        x = torch.flatten(x, 1) #(64, 12, 12) boyutundaki veriyi (9216) boyutuna düşürür

        X = self.fc1(x) #(9216) boyutundaki veriyi (128) boyutuna düşürür 
        x = F.relu(x)
        x = self.dropout2(x) #eğitim sırasında nöronların %50 sini rastgele kapatarak overfittingi önler

        cikti = self.fc2(x) #(128) boyutundaki veriyi (10) boyutuna düşürür, 0 dan 9 a kadar olan rakamları temsil eder
        return cikti 

if __name__ == "__main__": 
    model = RakamTanima()
    yanlis_resim = torch.randn(1, 1, 28, 28) #verilen boyutlarda rastgele bir resim oluşturur, (1, 1, 28, 28) boyutunda hayali bir resim oluşturur, 1. boyut batch size, 2. boyut kanal sayısı, 3. ve 4. boyut resim boyutu
    sonuc = model(yanlis_resim) #hayali resim modelden geçirilir
    print("Model başarıyla olışturuldu!") 
    print("Girdi Boyutu: ", yanlis_resim.shape) #yanlış resmin boyutunu ekrana yazdırır
    print("Çıktı Boyutu: ", sonuc.shape, "---("Bir resim için 0-9 arası rakam tahmini yapar")") #modelin çıktısının boyutunu ekrana yazdırır, 10 boyutlu çıktı verir, 0 dan 9 a kadar olan rakamları temsil eder

