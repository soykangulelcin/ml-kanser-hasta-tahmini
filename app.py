import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Meme Sağlığı Rehberi", layout="centered")

# Proje bilgileri
PROJE_SAHIBI = "Mete Gündoğdu"
KURUM = "Bartın Üniversitesi Meslek Yüksekokulu, Robotik ve Yapay Zeka Programı"
ILETISIM = "GitHub: github.com/meteefgd"

st.markdown("""
<style>
.block-container{max-width:960px;padding-top:1.5rem}
.hero{background:linear-gradient(135deg,#be185d 0%,#831843 100%);color:#fff;padding:1.7rem 1.9rem;border-radius:16px;margin-bottom:1rem}
.hero h1{margin:0;font-size:1.9rem;color:#fff;padding:0}
.hero p{margin:.5rem 0 0;font-size:1.02rem;opacity:.95}
.disc{background:#fff7ed;border-left:4px solid #f59e0b;padding:.75rem 1rem;border-radius:6px;color:#7c2d12;font-size:.93rem;margin-bottom:.6rem}
.card{border:1px solid rgba(128,128,128,.3);border-radius:12px;padding:.9rem 1.1rem;margin-bottom:.7rem}
.card b{color:#be185d}
.foot{text-align:center;font-size:.82rem;opacity:.75;margin-top:2rem;padding-top:1rem;border-top:1px solid rgba(128,128,128,.3)}
footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)

# Etiketler: 0 = Kötü huylu (kanser), 1 = İyi huylu
MODELLER = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
}


@st.cache_resource
def hazirla():
    v = load_breast_cancer()
    X = pd.DataFrame(v.data, columns=v.feature_names)
    y = pd.Series(v.target)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    egitilmis, satirlar = {}, []
    for ad, m in MODELLER.items():
        pipe = make_pipeline(StandardScaler(), m)
        pipe.fit(Xtr, ytr)
        p = pipe.predict(Xte)
        proba = pipe.predict_proba(Xte)[:, 1]
        egitilmis[ad] = pipe
        satirlar.append({
            "Model": ad,
            "Accuracy": accuracy_score(yte, p),
            "Kanser Recall": recall_score(yte, p, pos_label=0),
            "Kanser Precision": precision_score(yte, p, pos_label=0),
            "F1 (kanser)": f1_score(yte, p, pos_label=0),
            "AUC-ROC": roc_auc_score(yte, proba),
            "5-Fold CV Acc": cross_val_score(make_pipeline(StandardScaler(), m), X, y, cv=cv).mean(),
        })
    cms = {ad: confusion_matrix(yte, egitilmis[ad].predict(Xte)) for ad in MODELLER}
    return X, y, Xte, yte, egitilmis, pd.DataFrame(satirlar), cms


X, y, Xte, yte, egitilmis, sonuc_df, cms = hazirla()

st.markdown('''<div class="hero"><h1>Meme Sağlığı Rehberi</h1>
<p>Belirtileri tanıyın, taramanızı aksatmayın, gerekirse zamanında doktora başvurun.</p></div>''', unsafe_allow_html=True)

tab_belirti, tab_tarama, tab_ml, tab_yasal = st.tabs(
    ["Belirti Rehberi", "Tarama ve Kontrol", "Teknik Bilgiler", "Yasal Bilgiler"])
with tab_ml:
    tab1, tab2, tab3 = st.tabs(["Model Tahmini", "Model Karşılaştırma", "Veri Seti"])

# ---------------- TAHMİN ----------------
with tab1:
    st.info("**Teknik bilgi:** Bu bölüm projenin makine öğrenmesi kısmıdır. "
            "Modeller Breast Cancer Wisconsin veri setiyle eğitildi ve hücre çekirdeği ölçümlerini girdi olarak alır.")
    sol, sag = st.columns([1, 2])
    with sol:
        model_ad = st.selectbox("Model", list(MODELLER), index=0)
        idx = st.selectbox("Test setinden örnek hasta", range(len(Xte)),
                           format_func=lambda i: f"Hasta #{i}")
        if st.button("Bu hastanın değerlerini yükle"):
            for k, val in Xte.iloc[idx].items():
                st.session_state[k] = float(val)
            st.session_state["gercek"] = int(yte.iloc[idx])
        if st.button("Ortalamaya sıfırla"):
            for k in X.columns:
                st.session_state[k] = float(X[k].mean())
            st.session_state.pop("gercek", None)

    with st.expander("30 özelliği düzenle", expanded=False):
        cols = st.columns(3)
        girdi = []
        for i, k in enumerate(X.columns):
            lo, hi = float(X[k].min()), float(X[k].max())
            with cols[i % 3]:
                girdi.append(st.number_input(k, min_value=lo * 0.5, max_value=hi * 1.5,
                                             value=float(st.session_state.get(k, X[k].mean())),
                                             key=k, format="%.5f"))

    p1 = egitilmis[model_ad].predict_proba(pd.DataFrame([girdi], columns=X.columns))[0, 1]
    p0 = 1 - p1
    with sag:
        if p0 >= 0.5:
            st.error(f"### Kötü huylu (kanser) olasılığı yüksek: %{p0*100:.1f}")
        else:
            st.success(f"### İyi huylu olasılığı yüksek: %{p1*100:.1f}")
        st.progress(float(p0), text=f"Kötü huylu olasılığı: %{p0*100:.1f}")
        if "gercek" in st.session_state:
            g = "İyi huylu" if st.session_state["gercek"] == 1 else "Kötü huylu"
            st.info(f"Bu hastanın gerçek tanısı: **{g}**")
        st.caption("Not: Model, eğitim verisindeki dağılımın dışındaki değerlerde güvenilmez sonuç verir.")

# ---------------- KARŞILAŞTIRMA ----------------
with tab2:
    st.markdown("Tüm metrikler **gerçek model çıktılarından** hesaplanır "
                "(test seti: 114 hasta). **Kanser Recall** = gerçek kanserlerin kaçının yakalandığı; "
                "bu problemde en kritik metrik.")
    st.dataframe(sonuc_df.style.format({c: "{:.4f}" for c in sonuc_df.columns[1:]}),
                 width="stretch", hide_index=True)
    st.bar_chart(sonuc_df.set_index("Model")[["Accuracy", "Kanser Recall", "AUC-ROC", "5-Fold CV Acc"]])
    st.subheader("Confusion Matrix")
    c = st.columns(4)
    for col, (ad, cm) in zip(c, cms.items()):
        with col:
            st.markdown(f"**{ad}**")
            st.dataframe(pd.DataFrame(cm, index=["Gerçek: Kötü", "Gerçek: İyi"],
                                      columns=["Tahmin: Kötü", "Tahmin: İyi"]))

# ---------------- VERİ ----------------
with tab3:
    a, b, c3 = st.columns(3)
    a.metric("Hasta sayısı", len(X))
    b.metric("Kötü huylu", int((y == 0).sum()))
    c3.metric("İyi huylu", int((y == 1).sum()))
    st.dataframe(X.describe().T.round(3), width="stretch")

# ---------------- BELİRTİ REHBERİ ----------------
with tab_belirti:
    st.markdown('<div class="disc"><b>Önemli:</b> Bu araç teşhis koymaz ve doktor muayenesinin yerini tutmaz. '
                'Belirti seçmemiş olmanız kanser olmadığınız anlamına gelmez. Acil durumda 112\'yi arayın.</div>',
                unsafe_allow_html=True)
    yas = st.radio("Yaş grubunuz", ["40 yaş altı", "40–69 yaş", "70 yaş ve üzeri"], horizontal=True)
    onemli = ["Memede veya koltuk altında yeni bir sertlik / ele gelen kitle",
              "Meme ucundan kanlı veya kendiliğinden gelen akıntı",
              "Meme derisinde çukurlaşma, portakal kabuğu görünümü veya kızarıklık",
              "Memenin boyut ya da şeklinde belirgin, tek taraflı değişiklik"]
    diger = ["Meme ucunda yeni içe çekilme",
             "Meme ucu veya çevresinde pullanma / kabuklanma",
             "Düzelmeyen, tek bölgede süren meme ağrısı"]
    risk = ["Birinci derece akrabamda (anne, kız kardeş) meme kanseri öyküsü var",
            "Daha önce meme ile ilgili biyopsi / tedavi öyküm var"]
    k1, k2 = st.columns(2)
    with k1:
        st.markdown("**Fark ettiğiniz değişiklikler**")
        sec_onemli = [st.checkbox(x, key="o_" + x) for x in onemli]
        sec_diger = [st.checkbox(x, key="d_" + x) for x in diger]
    with k2:
        st.markdown("**Risk faktörleri**")
        sec_risk = [st.checkbox(x, key="r_" + x) for x in risk]

    ek = ("40–69 yaş arasındaki kadınlar için 2 yılda bir mamografi taraması KETEM, aile sağlığı merkezi ve "
          "Sağlıklı Hayat Merkezlerinde ücretsizdir.") if yas == "40–69 yaş" else ""
    st.divider()
    if any(sec_onemli):
        st.error("### Bir doktora başvurmanız önerilir\n"
                 "Seçtiğiniz belirtiler değerlendirme gerektirir. Mümkün olan en kısa sürede aile hekiminize, "
                 "genel cerrahi veya meme kliniğine başvurun. Bu belirtiler çoğu zaman iyi huylu nedenlerle de "
                 "olabilir; ayırt etmek doktorun işidir.")
    elif any(sec_diger) or any(sec_risk):
        st.warning("### Doktorunuza danışın\n"
                   "Seçtiğiniz belirti veya risk faktörü bir sağlık uzmanıyla konuşmaya değer. Yakın zamanda "
                   "randevu alın ve düzenli kontrollerinizi aksatmayın. " + ek)
    else:
        st.info("### Belirti seçilmedi\n"
                "Bu, risk olmadığı anlamına gelmez. Düzenli kendi kendine muayene ve tarama programını sürdürün; "
                "memelerinizde bir değişiklik fark ederseniz beklemeden doktora gidin. " + ek)

# ---------------- TARAMA VE KONTROL ----------------
with tab_tarama:
    st.markdown("Türkiye'de Sağlık Bakanlığı'nın Ulusal Kanser Tarama Programı kapsamındaki öneriler:")
    st.markdown("""
<div class="card"><b>Mamografi (tarama)</b><br>40–69 yaş arasındaki kadınlara <b>2 yılda bir</b>. KETEM, aile sağlığı merkezleri
ve Sağlıklı Hayat Merkezlerinde <b>ücretsiz</b> yapılır. Şikayeti olmayan kişilere yöneliktir.</div>
<div class="card"><b>Kendi kendine muayene</b><br><b>Ayda bir</b>. Önce aynada gözle, sonra elle yapılır.
Düzenli tarama ve doktor muayenesinin yerine geçmez.</div>
<div class="card"><b>Klinik meme muayenesi</b><br>Bir sağlık kuruluşunda hekim ya da sağlık profesyoneli tarafından <b>yılda bir</b>.</div>
<div class="card"><b>Ailede meme kanseri varsa</b><br>Birinci derece yakınında tanı alan biri varsa, kontroller genellikle
o kişinin tanı yaşından 5 yıl öncesinden başlatılır. Kişisel planınız için hekiminize danışın.</div>
<div class="card"><b>Normal sonuç garanti değildir</b><br>Mamografi tüm kanserleri saptayamaz. Normal bir sonuç, ileride
meme kanseri olmayacağınız anlamına gelmez; sonraki tarama tarihini aksatmayın.</div>
""", unsafe_allow_html=True)
    st.caption("Bilgiler Sağlık Bakanlığı tarama programı kaynaklarına dayanır. Güncel uygulama için en yakın "
               "KETEM veya aile sağlığı merkezine başvurun.")

# ---------------- YASAL BİLGİLER ----------------
with tab_yasal:
    with st.expander("Sorumluluk Reddi ve Tıbbi Uyarı", expanded=True):
        st.markdown("""
Bu uygulama **eğitim amaçlı bir ders projesidir**. İçerik yalnızca genel bilgilendirme amacıyla sunulur;
**tıbbi tavsiye, teşhis veya tedavi niteliği taşımaz** ve hekim muayenesinin yerini tutmaz.
Sağlığınızla ilgili kararları vermeden önce mutlaka bir sağlık profesyoneline danışın.
Uygulamanın sonuçlarına dayanarak tedaviyi geciktirmeyin, bırakmayın veya değiştirmeyin.
Acil durumlarda **112**'yi arayın.

Makine öğrenmesi modelleri yalnızca halka açık bir araştırma veri seti üzerinde eğitilmiştir, klinik olarak
doğrulanmamıştır ve gerçek hastalar için kullanılamaz. Belirti Rehberi basit kurallara dayanır;
yanlış negatif ve yanlış pozitif sonuçlar üretebilir.""")
    with st.expander("Gizlilik ve KVKK Aydınlatma Metni"):
        st.markdown(f"""
**Veri sorumlusu / proje sahibi:** {PROJE_SAHIBI} — {KURUM}

- Bu uygulama 6698 sayılı Kişisel Verilerin Korunması Kanunu (KVKK) kapsamında **kişisel veri toplamaz,
  kaydetmez ve üçüncü kişilerle paylaşmaz**. Kayıt, üyelik veya giriş yoktur.
- Seçtiğiniz belirtiler ve girdiğiniz değerler yalnızca oturum süresince geçici olarak kullanılır, bir veritabanına yazılmaz.
- Uygulama **Streamlit Community Cloud** üzerinde barındırılır. Barındırma sağlayıcısı, IP adresi ve tarayıcı bilgisi gibi
  teknik bağlantı kayıtlarını kendi gizlilik politikası çerçevesinde işleyebilir; bu işlemler proje sahibinin kontrolünde değildir.
- Uygulama kendisi reklam veya takip amaçlı çerez kullanmaz.
- Uygulamayı bir sağlık kaydı olarak kullanmayın; ad, T.C. kimlik no gibi kimlik bilgilerinizi girmeyin.
- KVKK m.11 kapsamındaki haklarınız ve sorularınız için: {ILETISIM}""")
    with st.expander("Kullanım Koşulları"):
        st.markdown(f"""
1. Uygulamayı kullanarak yukarıdaki sorumluluk reddini ve bu koşulları kabul etmiş sayılırsınız.
2. Uygulama **"olduğu gibi"** sunulur; içeriğin doğruluğu, eksiksizliği veya güncelliği konusunda garanti verilmez.
3. Proje sahibi, uygulamanın kullanımından veya kullanılamamasından doğabilecek doğrudan ya da dolaylı zararlardan, yasaların izin verdiği ölçüde sorumlu tutulamaz.
4. Uygulama ticari amaç taşımaz, reklam içermez ve herhangi bir sağlık ürünü veya hizmeti satmaz.
5. Uygulama içeriği izinsiz olarak tıbbi amaçla kullanılamaz.
6. Koşullar önceden haber verilmeden güncellenebilir. Uyuşmazlıklarda Türkiye Cumhuriyeti hukuku uygulanır.

© 2026 {PROJE_SAHIBI}""")
    with st.expander("Yazılımın Hukuki Niteliği"):
        st.markdown("""
Bu yazılım **tıbbi cihaz değildir** ve tıbbi cihaz mevzuatı kapsamında herhangi bir yetkili kurum tarafından
onaylanmamış veya tescil edilmemiştir. Hastalıkların teşhisi, tedavisi, izlenmesi veya önlenmesi amacıyla kullanılamaz.""")
    with st.expander("Veri Seti, Atıf ve Kaynaklar"):
        st.markdown("""
- **Veri seti:** Breast Cancer Wisconsin (Diagnostic) — W. H. Wolberg, W. N. Street, O. L. Mangasarian.
  UCI Machine Learning Repository, CC BY 4.0 lisansı ile paylaşılmaktadır
  (https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic). Uygulamada scikit-learn üzerinden kullanılmıştır.
- **Tarama bilgileri:** T.C. Sağlık Bakanlığı, Halk Sağlığı Genel Müdürlüğü — Ulusal Kanser Tarama Programı (KETEM).
- **Kullanılan kütüphaneler:** Streamlit, scikit-learn, pandas, NumPy.""")

st.markdown(f'<div class="foot">© 2026 {PROJE_SAHIBI} • {KURUM}<br>'
            'Eğitim amaçlı projedir, tıbbi teşhis aracı değildir. Acil durumda 112.</div>', unsafe_allow_html=True)
