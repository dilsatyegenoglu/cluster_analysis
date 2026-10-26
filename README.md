# 🌳 Biyolojik Bilimlerde Kümeleme Analizi

**Yüksek lisans dersi için etkileşimli, Türkçe öğretim uygulaması: numerik taksonomiden uzaklık ölçülerine, UPGMA ve Neighbor-Joining ağaçlarına**

© 2026 Emine Dilşat Yeğenoğlu & Tevfik Hasan Can. Tüm hakları saklıdır.

---

## Nedir?

Numerik taksonominin temel kavramlarını (OTU, karakter, veri matrisi), benzerlik ve uzaklık ölçülerini
(Öklid, Mahalanobis, Jaccard, basit eşleşme, Dice) ve uzaklık matrisinden adım adım ağaç kurmayı
(UPGMA, tek ve tam bağlantı, Neighbor-Joining) etkileşimli hesaplayıcılar, şemalar ve çözümlü
örneklerle anlatır. Bilgisayardan ve telefondan tarayıcıyla açılır.

## İçerik

| Bölüm | Konu ve etkileşimli öğeler |
|---|---|
| 1 · Numerik taksonomi ve veri matrisi | OTU, karakter (attribute), karakter durumu, karakter türleri ve kodlama, Adanson ilkesi, çalışma akışı, fenetik–kladistik, iki oyun |
| 2 · Uzaklık ve yakınlık | 0'a / 1'e yakın değerlerin anlamı, s → d dönüşümü, ölçülerin aralıkları, metrik koşulları, üçgen eşitsizliği denemesi |
| 3 · Öklid uzaklığı | İki karakterde çizimli hesap, Manhattan karşılaştırması, çok karakterli adım adım hesap, ham / z-skoru / aralık standartlaştırmasının ağaca etkisi |
| 4 · Mahalanobis uzaklığı | Korelasyon kaydırıcısıyla Öklid–Mahalanobis farkı, Iris versicolor verisiyle adım adım kovaryans, ters matris ve uzaklık |
| 5 · İkili veri | a-b-c-d tablosu, Jaccard / basit eşleşme / Dice hesaplayıcı, ortak sıfır sorunu grafiği, ISSR örnek matrisleri |
| 6 · UPGMA ve dendrogram | Adım adım matris güncelleme ve ağaç kurma (UPGMA, tek, tam bağlantı), dendrogram okuma, kesim düzeyi, kofenetik korelasyon |
| 7 · Neighbor-Joining | Satır toplamları, Q matrisi, dal uzunlukları, yeni düğüm uzaklıkları, köksüz ağaç, ağaçtaki yol ile özgün uzaklık karşılaştırması |
| 8 · Biyolojik veriler | Veri türüne göre ölçü seçimi; markör, morfoloji, ekspresyon, tür bolluğu ve Iris verileriyle kendi analizini yapma |
| 9 · Çözümlü örnekler | Sekiz konuda her tıklamada yeni sayılarla tam çözümlü problem |
| 10 · Genel tekrar | Hafıza kartları, sözlük, yöntemin tarihi, karışık sorular |

Her bölümün sonunda **10, 20 ya da 30 soruluk seçmeli alıştırma** vardır; hesap soruları her seferinde
yeni sayılarla üretilir. Yanıt seçilince kilitlenir; doğru/yanlış ve çözüm o zaman açılır.

K-ortalamalar (k-means) bölümü dersin ilerleyen haftalarında eklenecektir.

## Kurulum ve çalıştırma

```bash
pip install -r requirements.txt
streamlit run kume_streamlit.py
```

Linux / WSL'de kısaca: `bash baslat.sh`. Ardından tarayıcıda http://localhost:8501

## Yayına alma

GitHub reposuna `kume_streamlit.py` ve `requirements.txt` yüklenir; share.streamlit.io'da
"Create app" ile ana dosya olarak `kume_streamlit.py` seçilir.

## Veriler ve bilinen sınırlamalar

- **Gerçek veri:** Iris ölçümleri (Anderson 1935; Fisher 1936).
- **Öğretici (kurgusal) veriler:** ISSR bant matrisi, morfolojik ölçümler, gen ekspresyon profilleri ve tür bolluk tablosu yöntemi göstermek için uydurulmuştur; gerçek çeşit ya da tür verisi değildir. Uygulamada her birinin yanında belirtilmiştir.
- Bütün hesaplar (uzaklıklar, UPGMA, tek/tam bağlantı, NJ, kofenetik korelasyon) uygulamanın içinde yazılmıştır; UPGMA/tek/tam bağlantı birleşme düzeyleri ve kofenetik korelasyon SciPy ile, NJ dal uzunlukları Saitou–Nei'nin bilinen örneğiyle doğrulanmıştır.
- Eşit uzaklıklar varsa uygulama ilk bulduğu çifti birleştirir; başka programlar farklı ama eşdeğer bir ağaç çizebilir.
- Öğrenci yanıtları kaydedilmez.

## Telif ve kullanım

© 2026 Emine Dilşat Yeğenoğlu & Tevfik Hasan Can. Tüm hakları saklıdır. Ayrıntılar için
`LICENSE` dosyasına bakın.

İletişim ve izin talepleri: dilsat.yegenoglu@cbu.edu.tr
