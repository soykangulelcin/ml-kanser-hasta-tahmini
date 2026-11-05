[README (4).md](https://github.com/user-attachments/files/34424046/README.4.md)
# Meme Sağlığı Rehberi

Meme sağlığı farkındalığı için hazırlanmış bir web uygulaması ve bu uygulamanın arkasındaki makine öğrenmesi çalışması. Proje iki bölümden oluşur:

1. **Bilgilendirme arayüzü:** belirti rehberi, Türkiye'deki ulusal tarama programına dair özet bilgiler ve yasal bilgilendirme metinleri.
2. **Makine öğrenmesi çalışması:** Breast Cancer Wisconsin veri seti üzerinde dört sınıflandırma modelinin karşılaştırılması.

**Canlı uygulama:** https://ml-kanser-hasta-tahmini-gpsyaxynkgmudwzmfnfreq.streamlit.app/

> **Uyarı:** Bu proje eğitim amaçlı bir ders projesidir. Tıbbi tavsiye, teşhis veya tedavi niteliği taşımaz, tıbbi cihaz değildir ve klinik olarak doğrulanmamıştır. Sağlıkla ilgili kararlar için mutlaka bir sağlık profesyoneline başvurulmalıdır.

---

## İçindekiler

- [Uygulama Özellikleri](#uygulama-özellikleri)
- [Veri Seti](#veri-seti)
- [Yöntem](#yöntem)
- [Sonuçlar](#sonuçlar)
- [Sınırlılıklar](#sınırlılıklar)
- [Proje Yapısı](#proje-yapısı)
- [Kurulum ve Çalıştırma](#kurulum-ve-çalıştırma)
- [Yayınlama](#yayınlama)
- [Gizlilik](#gizlilik)
- [Kaynaklar ve Lisans](#kaynaklar-ve-lisans)
- [Yazar](#yazar)

---

## Uygulama Özellikleri

| Sekme | İçerik |
|---|---|
| Belirti Rehberi | Kullanıcının seçtiği belirti ve risk faktörlerine göre kural tabanlı yönlendirme (doktora başvuru önerisi seviyeleri) |
| Tarama ve Kontrol | T.C. Sağlık Bakanlığı Ulusal Kanser Tarama Programı kapsamındaki mamografi, kendi kendine muayene ve klinik muayene önerileri |
| Teknik Bilgiler | Model tahmini, dört modelin karşılaştırması (metrikler, confusion matrix) ve veri seti özeti |
| Yasal Bilgiler | Sorumluluk reddi, KVKK aydınlatma metni, kullanım koşulları, veri seti atfı |

Uygulama kişisel veri toplamaz, kaydetmez ve kullanıcı girişi gerektirmez.

## Veri Seti

- **Kaynak:** Breast Cancer Wisconsin (Diagnostic), `sklearn.datasets.load_breast_cancer` üzerinden
- **Boyut:** 569 örnek, 30 sayısal özellik (hücre çekirdeği ölçümleri), eksik değer yok
- **Etiketler:** `0` = kötü huylu (212), `1` = iyi huylu (357). Sınıf dağılımı hafif dengesizdir.

## Yöntem

- Veri %80 eğitim / %20 test olarak bölündü (`stratify=y`, `random_state=42`). Test kümesi 114 örnekten oluşur.
- Ölçekleme (`StandardScaler`) her modelle birlikte bir `Pipeline` içinde uygulandı. Böylece ölçekleme yalnızca eğitim verisinden öğrenilir ve veri sızıntısı oluşmaz.
- Karşılaştırılan modeller: Logistic Regression, Decision Tree, Random Forest (100 ağaç), KNN (k=5).
- Değerlendirme: test kümesi metrikleri ve ek olarak 5 katlı katmanlı çapraz doğrulama (`StratifiedKFold`).
- Kötü huylu sınıf pozitif sınıf olarak alındı. Kanser tespitinde bir vakayı kaçırmanın maliyeti yanlış alarmdan yüksek olduğu için ana metrik **kanser recall** değeridir.

## Sonuçlar

Aşağıdaki değerler kodun yeniden çalıştırılmasıyla doğrulanmış gerçek model çıktılarıdır.

| Model | Accuracy | Kanser Recall | Kanser Precision | F1 (kanser) | AUC-ROC | 5-Fold CV Acc |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9762 | 0.9762 | 0.9762 | 0.9954 | 0.9737 |
| Decision Tree | 0.9123 | 0.9286 | 0.8478 | 0.8864 | 0.9157 | 0.9104 |
| Random Forest | 0.9561 | 0.9286 | 0.9512 | 0.9398 | 0.9939 | 0.9561 |
| KNN (k=5) | 0.9561 | 0.9286 | 0.9512 | 0.9398 | 0.9788 | 0.9631 |

**Değerlendirme**

- Logistic Regression test kümesinde tüm metriklerde en iyi sonucu verdi. Ancak test kümesi 114 örnek olduğundan 1-2 örneklik farklar istatistiksel olarak anlamlı kabul edilmemelidir. Çapraz doğrulamada Logistic Regression, Random Forest ve KNN birbirine yakındır (%95,6 – %97,4).
- Decision Tree, kanser recall'da Random Forest ve KNN ile aynı değere sahiptir; ancak precision, F1, doğruluk ve AUC bakımından geride kalır. Budanmamış tek bir ağaç aşırı öğrenmeye yatkındır.
- Veri seti küçük ve temizdir. Gerçek klinik verilerde skorların düşmesi beklenir.

## Sınırlılıklar

- Hiperparametre optimizasyonu (ör. `GridSearchCV`) yapılmadı. Gelecek çalışma olarak eklenebilir.
- Modeller yalnızca halka açık, tek bir araştırma veri setiyle eğitildi. Harici bir veri setinde doğrulama yapılmadı.
- Model girdileri görüntü veya muayene verisi değil, hücre çekirdeği ölçümleridir. Bu nedenle son kullanıcıya doğrudan bir tanı aracı olarak sunulamaz.
- Belirti Rehberi basit kurallara dayanır. Yanlış negatif ve yanlış pozitif sonuçlar üretebilir.

## Proje Yapısı

```
.
├── app.py                  # Streamlit uygulaması
├── kanser_tahmini.ipynb    # Analiz ve model karşılaştırması
├── requirements.txt        # Bağımlılıklar
└── README.md
```

## Kurulum ve Çalıştırma

Python 3.9 veya üzeri önerilir.

```bash
git clone https://github.com/meteefgd/<repo-adı>.git
cd <repo-adı>
pip install -r requirements.txt
streamlit run app.py
```

Uygulama varsayılan olarak `http://localhost:8501` adresinde açılır.

## Yayınlama

Uygulama Streamlit Community Cloud üzerinde yayınlanmaktadır.

1. Dosyaları GitHub deposuna yükleyin.
2. https://share.streamlit.io adresinde GitHub hesabıyla giriş yapın ve **New app** seçeneğini kullanın.
3. Depoyu ve `app.py` dosyasını seçip **Deploy** düğmesine basın.

## Gizlilik

Uygulama 6698 sayılı KVKK kapsamında kişisel veri toplamaz ve saklamaz. Girilen değerler yalnızca oturum süresince geçici olarak kullanılır. Barındırma sağlayıcısının teknik bağlantı kayıtları kendi gizlilik politikasına tabidir. Ayrıntılar uygulamadaki **Yasal Bilgiler** sekmesinde yer almaktadır.

## Kaynaklar ve Lisans

- **Veri seti:** W. H. Wolberg, W. N. Street, O. L. Mangasarian, *Breast Cancer Wisconsin (Diagnostic)*. UCI Machine Learning Repository, CC BY 4.0. https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- **Tarama bilgileri:** T.C. Sağlık Bakanlığı, Halk Sağlığı Genel Müdürlüğü, Ulusal Kanser Tarama Programı (KETEM).
- **Kütüphaneler:** Streamlit, scikit-learn, pandas, NumPy.

Proje eğitim amaçlıdır ve ticari kullanıma yönelik değildir. Kaynak kod için bir yazılım lisansı belirtilmemiştir; kullanım veya yeniden dağıtım öncesinde yazarla iletişime geçilmesi önerilir.

## Yazar

**Mete Gündoğdu**  
Bartın Üniversitesi Meslek Yüksekokulu, Robotik ve Yapay Zeka Programı  
GitHub: [github.com/meteefgd](https://github.com/meteefgd)

**Soykan Gülelçin**  
Bartın Üniversitesi Meslek Yüksekokulu, Robotik ve Yapay Zeka Programı  
GitHub: [github.com/soykangulelcin](https://github.com/soykangulelcin)
