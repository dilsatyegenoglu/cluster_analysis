# -*- coding: utf-8 -*-
"""
Biyolojik Bilimlerde Kümeleme Analizi - Öğretici Streamlit Uygulaması
=====================================================================
© 2026 Emine Dilşat Yeğenoğlu & Tevfik Hasan Can. Tüm hakları saklıdır.

Yüksek lisans "Biyolojik Bilimlerde Kümeleme Analizi" dersi için etkileşimli öğretim
aracı: numerik taksonomi, OTU ve karakter, benzerlik ve uzaklık, Öklid ve Mahalanobis
uzaklıkları, ikili veri için Jaccard / basit eşleşme / Dice, uzaklık matrisi, UPGMA,
tek ve tam bağlantı, dendrogram okuma, kofenetik korelasyon, Neighbor-Joining ve
biyolojik veri türlerine göre ölçü seçimi. Bol çözümlü örnek; her konunun sonunda
10 / 20 / 30 seçmeli alıştırma ve açıklamalı çözüm.

Kurulum   : pip install -r requirements.txt
Çalıştırma: streamlit run kume_streamlit.py
"""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Kümeleme Analizi", page_icon="🌳", layout="wide")

TELIF = "© 2026 Emine Dilşat Yeğenoğlu & Tevfik Hasan Can. Tüm hakları saklıdır."

# Renkler
MAVI, PEMBE, YESIL, TURUNCU = "#3B6FB6", "#D9577A", "#3C9A5F", "#E8A33B"
MOR, GRI_K, SARI = "#8E5BB5", "#5F6B7A", "#F2C94C"
LACIVERT, TURKUAZ, MERCAN, ZEYTIN, ERIK = "#1F3A5F", "#1A9C9C", "#E4572E", "#6A8D2F", "#6B3FA0"
KUME_RENK = ["#1A9C9C", "#E4572E", "#6B3FA0", "#E8A33B", "#3B6FB6", "#6A8D2F", "#D9577A", "#5F6B7A"]
YAZI = 'font-family="Source Sans Pro, Segoe UI, Arial, sans-serif"'

st.markdown("""
<style>
.kutucuk {border-left:5px solid #3B6FB6; background:rgba(59,111,182,0.08);
          padding:10px 14px; border-radius:6px; margin:6px 0 10px 0;}
.uyari {border-left:5px solid #D9534F; background:rgba(217,83,79,0.08);
        padding:10px 14px; border-radius:6px; margin:6px 0 10px 0;}
.ipucu {border-left:5px solid #3C9A5F; background:rgba(60,154,95,0.08);
        padding:10px 14px; border-radius:6px; margin:6px 0 10px 0;}
.kartlar {display:flex; flex-wrap:wrap; gap:12px; margin:8px 0 14px 0;}
.kart {flex:1 1 220px; border:1px solid rgba(128,128,128,0.35); border-radius:10px;
       padding:12px 14px; background:rgba(128,128,128,0.05);}
.kart h4 {margin:0 0 6px 0; font-size:1.02rem;}
.kart p {margin:0; font-size:0.93rem;}




.zaman {border-left:4px solid #3B6FB6; padding:4px 0 4px 12px; margin:6px 0;}
.zaman b.yil {color:#3B6FB6; font-size:1.05rem;}
.flash {border:2px solid #3B6FB6; border-radius:14px; padding:26px; text-align:center;
        font-size:1.4rem; font-weight:700; margin:10px 0;}
.flash-cevap {font-size:1.05rem; font-weight:400; margin-top:12px;}
.adim-baslik {font-size:1.15rem; font-weight:700; color:#3B6FB6; margin-top:18px;
              border-bottom:2px solid rgba(59,111,182,0.3);}


.hero-alt {text-align:center; color:#1F3A5F; font-size:1.05rem; margin-top:-6px;}
.adim {border:1px solid rgba(26,156,156,0.45); border-radius:10px; padding:10px 14px; margin:8px 0;
       background:rgba(26,156,156,0.05);}
.adim b.no {color:#1A9C9C;}
.formul {font-family:monospace; font-size:0.95rem; background:rgba(128,128,128,0.08); border-radius:6px;
         padding:6px 10px; margin:4px 0; overflow-x:auto; white-space:nowrap;}
table.mat {border-collapse:collapse; margin:6px 0 12px 0; font-size:0.92rem;}
table.mat th, table.mat td {border:1px solid rgba(128,128,128,0.45); padding:4px 8px; text-align:center; min-width:38px;}
table.mat th {background:rgba(31,58,95,0.10);}
table.mat td.en {background:#FFE08A; font-weight:700; color:#111;}
table.mat td.yeni {background:rgba(26,156,156,0.18); font-weight:700;}
table.mat td.kosegen {color:#999;}
.mat-kap {overflow-x:auto;}
</style>
""", unsafe_allow_html=True)


def ipucu(metin):
    st.markdown(f'<div class="ipucu">{metin}</div>', unsafe_allow_html=True)


def kartlar(liste):
    """liste: [(başlık, metin), ...]"""
    icerik = "".join(f'<div class="kart"><h4>{b}</h4><p>{m}</p></div>' for b, m in liste)
    st.markdown(f'<div class="kartlar">{icerik}</div>', unsafe_allow_html=True)


def kutu(metin):
    st.markdown(f'<div class="kutucuk">{metin}</div>', unsafe_allow_html=True)


def uyari(metin):
    st.markdown(f'<div class="uyari">{metin}</div>', unsafe_allow_html=True)


def ciz(fig, yukseklik=380):
    fig.update_layout(height=yukseklik, margin=dict(l=40, r=20, t=50, b=105),
                      legend=dict(orientation="h", yanchor="top", y=-0.3, x=0))
    try:
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def tablo(df):
    try:
        st.dataframe(df, hide_index=True, width="stretch")
    except TypeError:
        st.dataframe(df, hide_index=True, use_container_width=True)


def rng():
    if "rng" not in st.session_state:
        st.session_state.rng = np.random.default_rng()
    return st.session_state.rng


SORU_SAYISI = 10


def _md(t):
    return ("md", t)


def _tx(t):
    return ("tex", t)


def _fmt(x, ondalik=4):
    if abs(x - round(x)) < 1e-9 and abs(x) >= 1:
        return f"{int(round(x))}"
    return f"{x:.{ondalik}f}".replace(".", ",")


def soru_sayi(metin, dogru, cozum, yanlis=(), ondalik=4, tol=None):
    """Sayısal soru; sık yapılan hatalardan türetilen yanlış seçeneklerle seçmeliye çevrilir."""
    return {"tip": "sayi", "metin": metin, "dogru": float(dogru), "yanlis": list(yanlis),
            "ondalik": ondalik, "cozum": cozum}


def soru_secim(r, metin, dogru, yanlislar, cozum):
    return {"tip": "secim", "metin": metin, "dogru": dogru, "yanlis": list(yanlislar),
            "cozum": cozum}


def _secmeliye_cevir(r, s):
    if s["tip"] == "secim":
        secenekler = [s["dogru"]] + s["yanlis"]
    else:
        x, od = s["dogru"], s["ondalik"]
        dogru = _fmt(x, od)
        olasilik = 0 < x < 1
        secenekler = [dogru]
        for v in s["yanlis"]:
            if v is None or not math.isfinite(v) or (v < 0 <= x) or (olasilik and v > 1):
                continue
            t = _fmt(v, od)
            if t not in secenekler:
                secenekler.append(t)
            if len(secenekler) == 4:
                break
        deneme = 0
        while len(secenekler) < 4 and deneme < 200:
            deneme += 1
            if abs(x - round(x)) < 1e-9 and abs(x) >= 1:
                adim_ = max(3, int(round(abs(x) * 0.25)))
                v = x + int(r.choice([-1, 1])) * int(r.integers(1, adim_ + 1))
            elif deneme > 40 or abs(x) < 10 ** (-od + 1):
                v = abs(x) + float(r.integers(5, 400)) * 10 ** (-od)
            else:
                v = x * (1 + float(r.choice([-1, 1])) * float(r.uniform(0.12, 0.6)))
            if (v < 0 <= x) or (olasilik and v >= 1):
                continue
            t = _fmt(v, od)
            if t not in secenekler:
                secenekler.append(t)
        s = dict(s, dogru=dogru)
    sira = r.permutation(len(secenekler))
    s["secenekler"] = [secenekler[i] for i in sira]
    return s


def _sec(r, liste):
    return liste[int(r.integers(0, len(liste)))]


def _kavram_sec(r, havuz, adet):
    idx = r.permutation(len(havuz))[:adet]
    return [havuz[i] for i in idx]


def _al_cevapla(k, i):
    v = st.session_state.get(f"{k}_c{i}")
    if v is not None and f"{k}_cevap{i}" not in st.session_state:
        st.session_state[f"{k}_cevap{i}"] = v


ADET_SECENEK = [10, 20, 30]


def _temizle(k):
    for i in range(1, max(ADET_SECENEK) + 1):
        for ek in ("c", "cevap", "kilit"):
            st.session_state.pop(f"{k}_{ek}{i}", None)


def _al_yenile(k, uretici):
    _temizle(k)
    st.session_state[k] = _uret(uretici, st.session_state.get(f"{k}_adet", SORU_SAYISI))


def _uret(uretici, adet=SORU_SAYISI):
    """Üreticiyi gerektiği kadar çağırıp metni tekrarlanmayan 'adet' soru toplar."""
    r = rng()
    sorular, gorulen = [], set()
    for _ in range(40):
        for s in uretici(r):
            if s["metin"] not in gorulen:
                gorulen.add(s["metin"])
                sorular.append(_secmeliye_cevir(r, s))
            if len(sorular) >= adet:
                return sorular
    return sorular


def alistirma_blogu(anahtar, baslik, uretici):
    st.divider()
    st.subheader(f"✍️ Alıştırmalar: {baslik}")
    st.caption("Her soruda bir seçenek işaretleyin. Yanıtınız kilitlenir; doğru ya da yanlış "
               "olduğu ve **adım adım çözüm** o zaman açılır. Önce kâğıt-kalemle hesaplayın!")
    k = f"al_{anahtar}"
    st.radio("Kaç soru?", ADET_SECENEK, horizontal=True, key=f"{k}_adet",
             on_change=_al_yenile, args=(k, uretici))
    if k not in st.session_state:
        st.session_state[k] = _uret(uretici, st.session_state.get(f"{k}_adet", SORU_SAYISI))
    sorular = st.session_state[k]
    dogru_say = cevap_say = 0
    for i, s in enumerate(sorular, 1):
        with st.container(border=True):
            st.markdown(f"**Soru {i} / {len(sorular)}.** {s['metin']}")
            secili = st.session_state.get(f"{k}_cevap{i}")
            if secili is None:
                st.radio("Seçiniz", s["secenekler"], index=None, key=f"{k}_c{i}",
                         label_visibility="collapsed", on_change=_al_cevapla, args=(k, i))
            else:
                cevap_say += 1
                st.radio("Yanıtınız", s["secenekler"], index=s["secenekler"].index(secili),
                         key=f"{k}_kilit{i}", disabled=True, label_visibility="collapsed")
                if secili == s["dogru"]:
                    dogru_say += 1
                    st.success("✅ Doğru!")
                else:
                    st.error(f"❌ Yanlış. Doğru yanıt: **{s['dogru']}**")
                with st.expander("📖 Nasıl hesaplanır?", expanded=True):
                    for tur, icerik in s["cozum"]:
                        if tur == "tex":
                            for parca in icerik.split(r"\qquad"):
                                st.latex(parca)
                        else:
                            st.markdown(icerik)
    st.progress(cevap_say / len(sorular),
                text=f"Yanıtlanan: {cevap_say} / {len(sorular)}  ·  Doğru: {dogru_say}")
    if cevap_say == len(sorular):
        oran = dogru_say / len(sorular)
        mesaj = (f"🎉 Tebrikler! {dogru_say}/{len(sorular)}" if oran >= 0.8 else
                 f"👍 {dogru_say}/{len(sorular)}: yanlışların çözümlerini tekrar inceleyin"
                 if oran >= 0.5 else
                 f"📚 {dogru_say}/{len(sorular)}: konuyu yukarıdan bir kez daha okuyup yeni "
                 "sorularla deneyin")
        (st.success if oran >= 0.8 else st.info if oran >= 0.5 else st.warning)(mesaj)
    st.button("🎲 Yeni sorular", key=f"{k}_yeni", on_click=_al_yenile, args=(k, uretici))


def svg_goster(svg):
    import base64
    veri = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    st.markdown(f'<div style="width:100%;margin:6px 0 14px 0;text-align:center">'
                f'<img src="data:image/svg+xml;base64,{veri}" style="width:100%;'
                f'max-width:1000px;background:white;border-radius:12px"/></div>',
                unsafe_allow_html=True)


def _yazi(x, y, t, boy=14, renk="#222", hiza="middle", kalin=False):
    k = ' font-weight="700"' if kalin else ""
    return (f'<text x="{x}" y="{y}" {YAZI} font-size="{boy}" fill="{renk}" '
            f'text-anchor="{hiza}"{k}>{t}</text>')


def _ok(x1, y1, x2, y2, renk="#555"):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{renk}" stroke-width="2.2" '
            f'marker-end="url(#okucu)"/>')


_OK_TANIM = ('<defs><marker id="okucu" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
             'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
             'fill="#555"/></marker></defs>')


# ===========================================================================
# HESAP ÇEKİRDEĞİ: uzaklıklar, benzerlik katsayıları, UPGMA / tek / tam bağlantı, NJ
# ===========================================================================
def _tr(x, od=2):
    """Türkçe ondalık gösterim (virgül)."""
    t = f"{x:.{od}f}".rstrip("0").rstrip(".") if od else f"{x:.0f}"
    if t in ("-0", ""):
        t = "0"
    tam, _, ond = t.partition(".")
    if abs(int(tam)) >= 10000:
        tam = f"{int(tam):,}".replace(",", ".")
    return tam + ("," + ond if ond else "")


def oklid(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.sqrt(np.sum((x - y) ** 2)))


def manhattan(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.sum(np.abs(x - y)))


def mahalanobis(x, y, S):
    f = np.asarray(x, float) - np.asarray(y, float)
    return float(np.sqrt(f @ np.linalg.inv(np.asarray(S, float)) @ f))


def z_standart(X):
    X = np.asarray(X, float)
    ss = X.std(axis=0, ddof=1)
    ss[ss == 0] = 1
    return (X - X.mean(axis=0)) / ss


def aralik_standart(X):
    X = np.asarray(X, float)
    a = X.max(axis=0) - X.min(axis=0)
    a[a == 0] = 1
    return (X - X.min(axis=0)) / a


def abcd(x, y):
    """İkili iki OTU için a (1-1), b (1-0), c (0-1), d (0-0)."""
    x, y = np.asarray(x, int), np.asarray(y, int)
    return (int(np.sum((x == 1) & (y == 1))), int(np.sum((x == 1) & (y == 0))),
            int(np.sum((x == 0) & (y == 1))), int(np.sum((x == 0) & (y == 0))))


def jaccard(x, y):
    a, b, c, _ = abcd(x, y)
    return a / (a + b + c) if a + b + c else 1.0


def basit_eslesme(x, y):
    a, b, c, d = abcd(x, y)
    return (a + d) / (a + b + c + d)


def dice(x, y):
    a, b, c, _ = abcd(x, y)
    return 2 * a / (2 * a + b + c) if a + b + c else 1.0


def pearson_r(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    xs, ys = x - x.mean(), y - y.mean()
    alt = math.sqrt(float(np.sum(xs ** 2) * np.sum(ys ** 2)))
    return float(np.sum(xs * ys) / alt) if alt else 0.0


def bray_curtis(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    t = float(np.sum(x + y))
    return float(np.sum(np.abs(x - y)) / t) if t else 0.0


OLCULER = {
    "Öklid": oklid,
    "Manhattan": manhattan,
    "Jaccard uzaklığı (1 − J)": lambda x, y: 1 - jaccard(x, y),
    "Basit eşleşme uzaklığı (1 − SM)": lambda x, y: 1 - basit_eslesme(x, y),
    "Dice uzaklığı (1 − D)": lambda x, y: 1 - dice(x, y),
    "Korelasyon uzaklığı (1 − r)": lambda x, y: 1 - pearson_r(x, y),
    "Bray–Curtis": bray_curtis,
}


def uzaklik_matrisi(X, olcu):
    X = np.asarray(X, float)
    n = len(X)
    f = OLCULER[olcu]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = f(X[i], X[j])
    return D


# ---------------------------------------------------------------------------
# Hiyerarşik kümeleme (adım adım)
# ---------------------------------------------------------------------------
YONTEM_AD = {"upgma": "UPGMA (ortalama bağlantı)", "tek": "Tek bağlantı (en yakın komşu)",
             "tam": "Tam bağlantı (en uzak komşu)"}


def hiyerarsik(D, adlar, yontem="upgma"):
    """Adım adım aglomeratif kümeleme.
    Döner: adimlar = [{'adlar', 'D', 'i', 'j', 'deger', 'yeni', 'formul'}], birlesme = [(u, v, h, boy)]
    birlesme dizinleri scipy düzenindedir (0..n-1 yapraklar, n.. iç düğümler)."""
    D = np.array(D, float)
    n = len(D)
    kume = [[i] for i in range(n)]
    ad = list(adlar)
    no = list(range(n))
    adimlar, birlesme = [], []
    sonraki = n
    while len(kume) > 1:
        m = len(kume)
        en, bi, bj = math.inf, 0, 1
        for i in range(m):
            for j in range(i + 1, m):
                if D[i, j] < en - 1e-12:
                    en, bi, bj = D[i, j], i, j
        ni, nj = len(kume[bi]), len(kume[bj])
        yeni_ad = f"({ad[bi]},{ad[bj]})"
        yeni_satir, formuller = [], []
        for k in range(m):
            if k in (bi, bj):
                continue
            a, b = D[bi, k], D[bj, k]
            if yontem == "upgma":
                v = (ni * a + nj * b) / (ni + nj)
                f = f"({ni}×{_tr(a, 3)} + {nj}×{_tr(b, 3)}) / {ni + nj} = {_tr(v, 3)}" if (ni > 1 or nj > 1) else \
                    f"({_tr(a, 3)} + {_tr(b, 3)}) / 2 = {_tr(v, 3)}"
            elif yontem == "tek":
                v = min(a, b)
                f = f"min({_tr(a, 3)}, {_tr(b, 3)}) = {_tr(v, 3)}"
            else:
                v = max(a, b)
                f = f"maks({_tr(a, 3)}, {_tr(b, 3)}) = {_tr(v, 3)}"
            yeni_satir.append(v)
            formuller.append((ad[k], f))
        adimlar.append({"adlar": list(ad), "D": D.copy(), "i": bi, "j": bj, "deger": float(en),
                        "yeni": yeni_ad, "formul": formuller})
        birlesme.append((no[bi], no[bj], float(en), ni + nj))
        kalan = [k for k in range(m) if k not in (bi, bj)]
        Dn = np.zeros((len(kalan) + 1, len(kalan) + 1))
        for x, k1 in enumerate(kalan):
            for y, k2 in enumerate(kalan):
                Dn[x + 1, y + 1] = D[k1, k2]
            Dn[0, x + 1] = Dn[x + 1, 0] = yeni_satir[x]
        D = Dn
        kume = [kume[bi] + kume[bj]] + [kume[k] for k in kalan]
        ad = [yeni_ad] + [ad[k] for k in kalan]
        no = [sonraki] + [no[k] for k in kalan]
        sonraki += 1
    return adimlar, birlesme


def yaprak_sirasi(birlesme, n):
    def ac(d):
        if d < n:
            return [d]
        u, v, _, _ = birlesme[d - n]
        return ac(int(u)) + ac(int(v))
    return ac(n + len(birlesme) - 1) if birlesme else list(range(n))


def kofenetik(birlesme, n):
    C = np.zeros((n, n))
    uye = {i: [i] for i in range(n)}
    for k, (u, v, h, _) in enumerate(birlesme):
        for a in uye[int(u)]:
            for b in uye[int(v)]:
                C[a, b] = C[b, a] = h
        uye[n + k] = uye[int(u)] + uye[int(v)]
    return C


def kofenetik_r(D, birlesme):
    n = len(D)
    C = kofenetik(birlesme, n)
    iu = np.triu_indices(n, 1)
    return pearson_r(np.asarray(D)[iu], C[iu])


def kesim_kumeleri(birlesme, n, esik):
    """Esikten küçük birleşmeleri uygulayarak küme etiketleri döndürür."""
    etiket = list(range(n))
    uye = {i: [i] for i in range(n)}
    for k, (u, v, h, _) in enumerate(birlesme):
        uye[n + k] = uye[int(u)] + uye[int(v)]
        if h <= esik + 1e-12:
            for a in uye[n + k]:
                etiket[a] = n + k
    tekil = {e: i + 1 for i, e in enumerate(dict.fromkeys(etiket))}
    return [tekil[e] for e in etiket]


# ---------------------------------------------------------------------------
# Neighbor-Joining (adım adım)
# ---------------------------------------------------------------------------
def komsu_birlestirme(D, adlar):
    """Saitou ve Nei (1987). Döner: adimlar, kenarlar [(düğüm1, düğüm2, uzunluk)]"""
    D = np.array(D, float)
    ad = list(adlar)
    adimlar, kenarlar = [], []
    sayac = 1
    while len(ad) > 2:
        n = len(ad)
        r = D.sum(axis=1)
        Q = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    Q[i, j] = (n - 2) * D[i, j] - r[i] - r[j]
        en, bi, bj = math.inf, 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                if Q[i, j] < en - 1e-12:
                    en, bi, bj = Q[i, j], i, j
        di = D[bi, bj] / 2 + (r[bi] - r[bj]) / (2 * (n - 2))
        dj = D[bi, bj] - di
        u = f"U{sayac}"
        sayac += 1
        yeni = []
        for k in range(n):
            if k not in (bi, bj):
                yeni.append((D[bi, k] + D[bj, k] - D[bi, bj]) / 2)
        adimlar.append({"adlar": list(ad), "D": D.copy(), "r": r.copy(), "Q": Q.copy(), "i": bi, "j": bj,
                        "Qmin": float(en), "di": float(di), "dj": float(dj), "u": u,
                        "yeni": [(ad[k], v) for k, v in zip([k for k in range(n) if k not in (bi, bj)], yeni)]})
        kenarlar.append((ad[bi], u, float(di)))
        kenarlar.append((ad[bj], u, float(dj)))
        kalan = [k for k in range(n) if k not in (bi, bj)]
        Dn = np.zeros((len(kalan) + 1, len(kalan) + 1))
        for x, k1 in enumerate(kalan):
            for y, k2 in enumerate(kalan):
                Dn[x + 1, y + 1] = D[k1, k2]
            Dn[0, x + 1] = Dn[x + 1, 0] = yeni[x]
        D = Dn
        ad = [u] + [ad[k] for k in kalan]
    kenarlar.append((ad[0], ad[1], float(D[0, 1])))
    adimlar.append({"adlar": list(ad), "D": D.copy(), "son": True})
    return adimlar, kenarlar


def agac_yol_uzakligi(kenarlar, a, b):
    komsu = {}
    for x, y, l in kenarlar:
        komsu.setdefault(x, []).append((y, l))
        komsu.setdefault(y, []).append((x, l))
    yigin = [(a, 0.0, None)]
    while yigin:
        d, top, ata = yigin.pop()
        if d == b:
            return top
        for k, l in komsu[d]:
            if k != ata:
                yigin.append((k, top + l, d))
    return math.nan


# ===========================================================================
# ÇİZİMLER (SVG)
# ===========================================================================
def svg_ac(w, h, arka="white"):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">'
            f'{_OK_TANIM}<rect x="0" y="0" width="{w}" height="{h}" fill="{arka}"/>')


def svg_hero():
    w, h = 1000, 380
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
         '<defs><linearGradient id="gk" x1="0" y1="0" x2="1" y2="1">'
         '<stop offset="0" stop-color="#0E2238"/><stop offset="1" stop-color="#1F3A5F"/></linearGradient>'
         '<radialGradient id="pa" cx="0.5" cy="0.5" r="0.5"><stop offset="0" stop-color="#ffffff" stop-opacity="0.9"/>'
         '<stop offset="1" stop-color="#ffffff" stop-opacity="0"/></radialGradient></defs>',
         f'<rect width="{w}" height="{h}" rx="18" fill="url(#gk)"/>']
    # Sağ: nokta bulutu, kümeler renkleniyor
    r = np.random.default_rng(7)
    merkez = [(640, 170), (820, 165), (720, 290), (895, 285)]
    for k, (cx, cy) in enumerate(merkez):
        renk = KUME_RENK[k]
        s.append(f'<circle cx="{cx}" cy="{cy}" r="58" fill="{renk}" opacity="0">'
                 f'<animate attributeName="opacity" values="0;0;0.18;0.18;0" keyTimes="0;0.35;0.5;0.9;1" dur="8s" repeatCount="indefinite"/></circle>')
        for _ in range(9):
            x, y = cx + r.normal(0, 22), cy + r.normal(0, 20)
            s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="#cfd8e3">'
                     f'<animate attributeName="fill" values="#cfd8e3;#cfd8e3;{renk};{renk};#cfd8e3" keyTimes="0;0.35;0.5;0.9;1" '
                     f'dur="8s" repeatCount="indefinite"/></circle>')
    # Sol: büyüyen dendrogram
    yap = [70, 110, 150, 190, 230, 270, 310]
    x0 = 60
    birles = [((0, 1), 120), ((2, 3), 150), ((4, 5), 135), ((6,), 0)]
    cizgi = []
    yy = {}
    for i, y in enumerate(yap):
        cizgi.append((x0, y, x0 + 40, y))
        yy[i] = y
    # elle kurulmuş ağaç
    def kol(xa, ya, xb, yb):
        cizgi.append((xa, ya, xb, ya))
    kol(x0 + 40, 70, x0 + 120, 0); kol(x0 + 40, 110, x0 + 120, 0)
    cizgi.append((x0 + 120, 70, x0 + 120, 110))
    kol(x0 + 40, 150, x0 + 150, 0); kol(x0 + 40, 190, x0 + 150, 0)
    cizgi.append((x0 + 150, 150, x0 + 150, 190))
    kol(x0 + 40, 230, x0 + 110, 0); kol(x0 + 40, 270, x0 + 110, 0)
    cizgi.append((x0 + 110, 230, x0 + 110, 270))
    kol(x0 + 110, 250, x0 + 190, 0); kol(x0 + 40, 310, x0 + 190, 0)
    cizgi.append((x0 + 190, 250, x0 + 190, 310))
    kol(x0 + 120, 90, x0 + 230, 0); kol(x0 + 150, 170, x0 + 230, 0)
    cizgi.append((x0 + 230, 90, x0 + 230, 170))
    kol(x0 + 230, 130, x0 + 300, 0); kol(x0 + 190, 280, x0 + 300, 0)
    cizgi.append((x0 + 300, 130, x0 + 300, 280))
    cizgi.append((x0 + 300, 205, x0 + 340, 205))
    s.append('<g transform="translate(40,118) scale(0.78)">')
    for k, (a, b, c, d) in enumerate(cizgi):
        uz = abs(c - a) + abs(d - b)
        bas = 0.04 * k
        s.append(f'<line x1="{a}" y1="{b}" x2="{c}" y2="{d}" stroke="#F2C94C" stroke-width="3.2" stroke-linecap="round" '
                 f'stroke-dasharray="{uz}" stroke-dashoffset="{uz}">'
                 f'<animate attributeName="stroke-dashoffset" values="{uz};{uz};0;0;{uz}" keyTimes="0;{bas:.2f};{bas + 0.12:.2f};0.92;1" '
                 f'dur="8s" repeatCount="indefinite"/></line>')
    renkler = [KUME_RENK[0], KUME_RENK[0], KUME_RENK[1], KUME_RENK[1], KUME_RENK[2], KUME_RENK[2], KUME_RENK[3]]
    for i, y in enumerate(yap):
        s.append(f'<circle cx="{x0}" cy="{y}" r="9" fill="{renkler[i]}" stroke="#fff" stroke-width="2"/>')
    s.append('</g>')
    s.append(f'<text x="{w / 2 + 5}" y="64" {YAZI} font-size="44" font-weight="800" fill="#F2C94C" text-anchor="middle" '
             f'letter-spacing="3">KÜMELEME ANALİZİ</text>')
    s.append(f'<text x="{w / 2 + 5}" y="96" {YAZI} font-size="17" fill="#cfd8e3" text-anchor="middle" letter-spacing="2">'
             f'BENZERLİKTEN AĞACA · BİYOLOJİK BİLİMLERDE</text>')
    s.append('</svg>')
    return "".join(s)


# ---------------------------------------------------------------------------
def _guzel_tik(enb):
    if enb <= 0:
        return [0, 1]
    ham = enb / 5
    us = 10 ** math.floor(math.log10(ham))
    adim = min((m * us for m in (1, 2, 2.5, 5, 10) if m * us >= ham), default=us * 10)
    return [k * adim for k in range(int(enb / adim + 1e-9) + 1)]


def svg_dendrogram(birlesme, adlar, adim=None, kesim=None, baslik="", eksen="Uzaklık", olcek_max=None, ultra_yarim=False):
    """Yatay dendrogram: yapraklar solda, uzaklık sağa doğru artar.
    adim: yalnız ilk 'adim' birleşmeyi çiz (son çizilen vurgulanır). kesim: kesme düzeyi."""
    n = len(adlar)
    vurgu_var = adim is not None and adim < len(birlesme) + 1 and adim is not None
    if adim is None:
        adim = len(birlesme)
    sira = yaprak_sirasi(birlesme, n) if birlesme else list(range(n))
    satir = 40
    sol, sag, ust = 140, 90, 46 if baslik else 26
    w = 560
    h = ust + satir * n + 64
    hmax = olcek_max or (max([b[2] for b in birlesme]) if birlesme else 1) or 1
    if ultra_yarim:
        hmax = hmax / 2
    def X(v):
        return sol + (w - sol - sag) * (v / hmax if not ultra_yarim else (v / 2) / hmax)
    yer = {d: ust + satir * (k + 0.5) for k, d in enumerate(sira)}
    xk = {d: sol for d in range(n)}
    etiket = None
    if kesim is not None and birlesme:
        etiket = kesim_kumeleri(birlesme[:adim], n, kesim)
    s = [svg_ac(w, h)]
    if baslik:
        s.append(_yazi(w / 2, 28, baslik, 21, LACIVERT, kalin=True))
    # eksen
    tik = _guzel_tik(olcek_max or (max([b[2] for b in birlesme]) if birlesme else 1))
    ye = ust + satir * n + 8
    s.append(f'<line x1="{sol}" y1="{ye}" x2="{w - sag}" y2="{ye}" stroke="#888" stroke-width="1.2"/>')
    for t in tik:
        xx = X(t)
        s.append(f'<line x1="{xx}" y1="{ye}" x2="{xx}" y2="{ye + 5}" stroke="#888"/>')
        s.append(_yazi(xx, ye + 20, _tr(float(t / 2 if ultra_yarim else t), 3), 15.6, "#555"))
        s.append(f'<line x1="{xx}" y1="{ust}" x2="{xx}" y2="{ye}" stroke="#eee" stroke-width="1"/>')
    s.append(_yazi((sol + w - sag) / 2, ye + 40, eksen, 16.9, "#555"))
    for k in range(adim):
        u, v, hh, _ = birlesme[k]
        u, v = int(u), int(v)
        x = X(hh)
        son = vurgu_var and k == adim - 1
        renk = MERCAN if son else LACIVERT
        kal = 3.6 if son else 2.4
        for d in (u, v):
            s.append(f'<line x1="{xk[d]}" y1="{yer[d]}" x2="{x}" y2="{yer[d]}" stroke="{renk}" stroke-width="{kal}"/>')
        s.append(f'<line x1="{x}" y1="{yer[u]}" x2="{x}" y2="{yer[v]}" stroke="{renk}" stroke-width="{kal}"/>')
        if son:
            s.append(_yazi(x + 6, (yer[u] + yer[v]) / 2 - 4, _tr(hh / 2 if ultra_yarim else hh, 3), 15.6, MERCAN, "start", True))
        yer[n + k] = (yer[u] + yer[v]) / 2
        xk[n + k] = x
    for d in range(n):
        renk = KUME_RENK[(etiket[d] - 1) % len(KUME_RENK)] if etiket else "#333"
        s.append(f'<circle cx="{sol}" cy="{yer[d]}" r="6.5" fill="{renk}"/>')
        s.append(_yazi(sol - 12, yer[d] + 5, adlar[d], 18.2, renk, "end", True))
    if kesim is not None:
        xx = X(kesim)
        s.append(f'<line x1="{xx}" y1="{ust - 6}" x2="{xx}" y2="{ye}" stroke="{MERCAN}" stroke-width="2" stroke-dasharray="6 4"/>')
        s.append(_yazi(xx, ust - 10 if ust > 30 else ust + 4, "kesim", 15.6, MERCAN))
    s.append("</svg>")
    return "".join(s)


# ---------------------------------------------------------------------------
def svg_nj_agac(kenarlar, yapraklar, baslik="", vurgu=()):
    """Köksüz ağaç, eşit açı yerleşimi. Negatif dal uzunlukları 0 olarak çizilir."""
    komsu = {}
    for a, b, l in kenarlar:
        komsu.setdefault(a, []).append((b, max(l, 0.0)))
        komsu.setdefault(b, []).append((a, max(l, 0.0)))
    ic = [d for d in komsu if d not in yapraklar]
    kok = ic[-1] if ic else yapraklar[0]

    def yaprak_say(d, ata):
        if d in yapraklar:
            return 1
        return sum(yaprak_say(c, d) for c, _ in komsu[d] if c != ata)

    poz = {kok: (0.0, 0.0)}
    ata_of = {}

    def yerlestir(d, ata, a0, a1):
        cocuk = [(c, l) for c, l in komsu[d] if c != ata]
        top = sum(yaprak_say(c, d) for c, _ in cocuk)
        a = a0
        for c, l in cocuk:
            pay = (a1 - a0) * yaprak_say(c, d) / top
            orta = a + pay / 2
            x0, y0 = poz[d]
            poz[c] = (x0 + l * math.cos(orta), y0 + l * math.sin(orta))
            ata_of[c] = orta
            yerlestir(c, d, a, a + pay)
            a += pay

    yerlestir(kok, None, 0, 2 * math.pi)
    xs = [p[0] for p in poz.values()]
    ys = [p[1] for p in poz.values()]
    w, h = 720, 460
    pay = 70
    ust = 40 if baslik else 10
    genis = max(max(xs) - min(xs), 1e-9)
    yuks = max(max(ys) - min(ys), 1e-9)
    k = min((w - 2 * pay) / genis, (h - ust - 2 * pay) / yuks)
    cx = (w - k * (max(xs) + min(xs))) / 2
    cy = ust + (h - ust - k * (max(ys) + min(ys))) / 2

    def P(d):
        return cx + k * poz[d][0], cy + k * poz[d][1]

    s = [svg_ac(w, h)]
    if baslik:
        s.append(_yazi(w / 2, 26, baslik, 21, LACIVERT, kalin=True))
    for a, b, l in kenarlar:
        (x1, y1), (x2, y2) = P(a), P(b)
        renk = MERCAN if (a in vurgu or b in vurgu) else LACIVERT
        s.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{renk}" stroke-width="3" stroke-linecap="round"/>')
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        s.append(f'<rect x="{mx - 18:.1f}" y="{my - 13:.1f}" width="46" height="21" rx="4" fill="white" opacity="0.85"/>')
        s.append(_yazi(mx, my + 3, _tr(l, 2), 15, "#555"))
    for d in poz:
        x, y = P(d)
        if d in yapraklar:
            aci = ata_of.get(d, 0.0)
            ox, oy = math.cos(aci) * 24, math.sin(aci) * 24
            s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{TURKUAZ}"/>')
            s.append(_yazi(x + ox, y + oy + 5, d, 18.8, "#222", kalin=True))
        else:
            s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="#999"/>')
            s.append(_yazi(x + 8, y + 16, d, 12.5, "#aaa", "start"))
    s.append("</svg>")
    return "".join(s)


# ---------------------------------------------------------------------------
def svg_ucgen(dab, dbc, dac):
    w, h = 640, 300
    ok = dac <= dab + dbc + 1e-12 and dab <= dac + dbc + 1e-12 and dbc <= dab + dac + 1e-12
    A, B, C = (110, 240), (320, 70), (530, 240)
    s = [svg_ac(w, h)]
    renk_ac = YESIL if ok else MERCAN
    s.append(f'<line x1="{A[0]}" y1="{A[1]}" x2="{B[0]}" y2="{B[1]}" stroke="{LACIVERT}" stroke-width="4"/>')
    s.append(f'<line x1="{B[0]}" y1="{B[1]}" x2="{C[0]}" y2="{C[1]}" stroke="{LACIVERT}" stroke-width="4"/>')
    s.append(f'<line x1="{A[0]}" y1="{A[1]}" x2="{C[0]}" y2="{C[1]}" stroke="{renk_ac}" stroke-width="5"/>')
    for p, ad in ((A, "A"), (B, "B"), (C, "C")):
        s.append(f'<circle cx="{p[0]}" cy="{p[1]}" r="20" fill="{TURKUAZ}"/>' + _yazi(p[0], p[1] + 6, ad, 25.2, "white", kalin=True))
    s.append(_yazi(195, 145, f"d(A,B) = {_tr(dab)}", 21, LACIVERT, "end", True))
    s.append(_yazi(445, 145, f"d(B,C) = {_tr(dbc)}", 21, LACIVERT, "start", True))
    s.append(_yazi(320, 284, f"d(A,C) = {_tr(dac)}", 21, renk_ac, kalin=True))
    s.append(_yazi(320, 26, f"d(A,C) = {_tr(dac)}  {'≤' if dac <= dab + dbc + 1e-12 else '&gt;'}  d(A,B) + d(B,C) = {_tr(dab + dbc)}",
                   20, renk_ac, kalin=True))
    s.append("</svg>")
    return "".join(s)


def svg_abcd(a, b, c, d, adx="OTU i", ady="OTU j"):
    w, h = 540, 300
    s = [svg_ac(w, h)]
    x0, y0, k = 200, 70, 120
    s.append(_yazi(x0 + k, 30, ady, 19, LACIVERT, kalin=True))
    s.append(_yazi(x0 + k / 2, 58, "1", 18, "#444", kalin=True) + _yazi(x0 + 1.5 * k, 58, "0", 18, "#444", kalin=True))
    s.append(_yazi(110, y0 + k * 0.85, adx, 19.2, LACIVERT, kalin=True))
    s.append(_yazi(x0 - 18, y0 + k / 2 + 5, "1", 18, "#444", kalin=True) + _yazi(x0 - 18, y0 + 1.5 * k + 5, "0", 18, "#444", kalin=True))
    hucre = [(0, 0, "a", a, YESIL, "ortak var"), (1, 0, "b", b, TURUNCU, "yalnız i'de"),
             (0, 1, "c", c, TURUNCU, "yalnız j'de"), (1, 1, "d", d, GRI_K, "ortak yok")]
    for i, j, harf, deg, renk, alt in hucre:
        x, y = x0 + i * k, y0 + j * k
        s.append(f'<rect x="{x}" y="{y}" width="{k}" height="{k * 0.82}" rx="8" fill="{renk}" opacity="0.18" stroke="{renk}" stroke-width="2"/>')
        s.append(_yazi(x + k / 2, y + 40, f"{harf} = {deg}", 26.4, renk, kalin=True))
        s.append(_yazi(x + k / 2, y + 66, alt, 15.6, "#444"))
    s.append("</svg>")
    return "".join(s)


def svg_veri_matrisi():
    w, h = 820, 300
    s = [svg_ac(w, h)]
    x0, y0, cw, ch = 210, 70, 80, 36
    kar = ["Karakter 1", "Karakter 2", "…", "Karakter t"]
    otu = ["OTU 1", "OTU 2", "OTU 3", "…", "OTU n"]
    s.append(_yazi(x0 + 2 * cw, 30, "KARAKTERLER (attribute, değişken) →  t sütun", 15, MERCAN, kalin=True))
    s.append(f'<text x="40" y="{y0 + 2.5 * ch}" {YAZI} font-size="15" fill="{TURKUAZ}" font-weight="700" '
             f'transform="rotate(-90 40 {y0 + 2.5 * ch})" text-anchor="middle">OTU'"'"'lar →  n satır</text>')
    for j, k in enumerate(kar):
        s.append(f'<rect x="{x0 + j * cw}" y="{y0 - ch}" width="{cw}" height="{ch}" fill="{MERCAN}" opacity="0.15" stroke="#999"/>')
        s.append(_yazi(x0 + j * cw + cw / 2, y0 - ch / 2 + 5, k, 12, "#333", kalin=True))
    ornek = [["1", "3,4", "…", "siyah"], ["0", "2,9", "…", "beyaz"], ["1", "3,6", "…", "siyah"], ["…", "…", "…", "…"], ["0", "4,1", "…", "pembe"]]
    for i, o in enumerate(otu):
        s.append(f'<rect x="{x0 - 110}" y="{y0 + i * ch}" width="110" height="{ch}" fill="{TURKUAZ}" opacity="0.15" stroke="#999"/>')
        s.append(_yazi(x0 - 55, y0 + i * ch + ch / 2 + 5, o, 13, "#333", kalin=True))
        for j in range(4):
            s.append(f'<rect x="{x0 + j * cw}" y="{y0 + i * ch}" width="{cw}" height="{ch}" fill="white" stroke="#999"/>')
            s.append(_yazi(x0 + j * cw + cw / 2, y0 + i * ch + ch / 2 + 5, ornek[i][j], 13, "#333"))
    s.append(_ok(x0 + 4 * cw + 20, y0 + 2.5 * ch, x0 + 4 * cw + 90, y0 + 2.5 * ch))
    s.append(f'<rect x="{x0 + 4 * cw + 96}" y="{y0 + 1.6 * ch}" width="150" height="{ch * 1.8}" rx="8" fill="{LACIVERT}" opacity="0.1" stroke="{LACIVERT}"/>')
    s.append(_yazi(x0 + 4 * cw + 171, y0 + 2.4 * ch, "Karakter durumu:", 12, LACIVERT, kalin=True))
    s.append(_yazi(x0 + 4 * cw + 171, y0 + 3.0 * ch, "hücredeki değer", 12, LACIVERT))
    s.append("</svg>")
    return "".join(s)


NT_ADIM = [
    ("OTU'ların seçimi", "Sınıflandırılacak birimler: birey, çeşit, popülasyon, tür"),
    ("Karakterlerin seçimi", "Olabildiğince çok, eşit ağırlıklı karakter"),
    ("Kodlama ve veri matrisi", "n OTU × t karakter; ikili, sıralı, nicel kodlama"),
    ("Standartlaştırma", "Nicel karakterleri ortak ölçeğe getirme (gerekirse)"),
    ("Benzerlik / uzaklık", "Her OTU çifti için katsayı: Jaccard, SM, Öklid, Gower..."),
    ("Kümeleme", "Uzaklık matrisinden UPGMA, tek/tam bağlantı ya da NJ"),
    ("Fenogram ve yorum", "Dendrogram, kofenetik korelasyon, grupların yorumu"),
]


def svg_nt_akis(sec):
    w = 760
    h = 60 + 62 * len(NT_ADIM)
    s = [svg_ac(w, h)]
    s.append(_yazi(w / 2, 30, "Numerik taksonomide çalışma akışı", 17, LACIVERT, kalin=True))
    for i, (b, a) in enumerate(NT_ADIM):
        y = 50 + i * 62
        aktif = i == sec
        renk = MERCAN if aktif else TURKUAZ
        s.append(f'<rect x="80" y="{y}" width="600" height="48" rx="10" fill="{renk}" opacity="{0.9 if aktif else 0.14}" '
                 f'stroke="{renk}" stroke-width="2"/>')
        s.append(f'<circle cx="80" cy="{y + 24}" r="18" fill="{renk}"/>' + _yazi(80, y + 30, str(i + 1), 16, "white", kalin=True))
        s.append(_yazi(110, y + 21, b, 15, "white" if aktif else LACIVERT, "start", True))
        s.append(_yazi(110, y + 40, a, 12.5, "white" if aktif else "#444", "start"))
        if i < len(NT_ADIM) - 1:
            s.append(f'<line x1="380" y1="{y + 48}" x2="380" y2="{y + 62}" stroke="#999" stroke-width="2"/>')
    s.append("</svg>")
    return "".join(s)


def svg_oklid_2d(p, q, adlar=("A", "B"), eksen=("x₁", "x₂")):
    w, h = 560, 400
    s = [svg_ac(w, h)]
    ox, oy, k = 70, 340, 26
    xm = max(10, p[0], q[0]) + 1
    ym = max(10, p[1], q[1]) + 1
    k = min(440 / xm, 270 / ym)
    s.append(f'<line x1="{ox}" y1="{oy}" x2="{ox + k * xm}" y2="{oy}" stroke="#888" stroke-width="1.5"/>')
    s.append(f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy - k * ym}" stroke="#888" stroke-width="1.5"/>')
    for t in range(0, int(xm) + 1):
        s.append(_yazi(ox + k * t, oy + 18, str(t), 14.3, "#777"))
    for t in range(0, int(ym) + 1):
        s.append(_yazi(ox - 10, oy - k * t + 4, str(t), 14.3, "#777", "end"))
    s.append(_yazi(ox + k * xm, oy + 34, eksen[0], 18.2, "#444", "end", True))
    s.append(_yazi(ox - 8, oy - k * ym - 8, eksen[1], 18.2, "#444", "start", True))
    P = (ox + k * p[0], oy - k * p[1])
    Q = (ox + k * q[0], oy - k * q[1])
    s.append(f'<line x1="{P[0]}" y1="{P[1]}" x2="{Q[0]}" y2="{P[1]}" stroke="{TURUNCU}" stroke-width="3" stroke-dasharray="7 4"/>')
    s.append(f'<line x1="{Q[0]}" y1="{P[1]}" x2="{Q[0]}" y2="{Q[1]}" stroke="{TURUNCU}" stroke-width="3" stroke-dasharray="7 4"/>')
    s.append(f'<line x1="{P[0]}" y1="{P[1]}" x2="{Q[0]}" y2="{Q[1]}" stroke="{MERCAN}" stroke-width="4"/>')
    s.append(_yazi((P[0] + Q[0]) / 2, P[1] + (18 if Q[1] < P[1] else -8), f"|Δx₁| = {_tr(abs(q[0] - p[0]))}", 16.9, TURUNCU, kalin=True))
    s.append(_yazi(Q[0] + 8, (P[1] + Q[1]) / 2, f"|Δx₂| = {_tr(abs(q[1] - p[1]))}", 16.9, TURUNCU, "start", True))
    d = math.hypot(q[0] - p[0], q[1] - p[1])
    s.append(_yazi((P[0] + Q[0]) / 2 - 12, (P[1] + Q[1]) / 2 - 10, f"d = {_tr(d, 3)}", 19.5, MERCAN, "end", True))
    for (x, y), ad in ((P, adlar[0]), (Q, adlar[1])):
        s.append(f'<circle cx="{x}" cy="{y}" r="9" fill="{TURKUAZ}" stroke="white" stroke-width="2"/>')
        s.append(_yazi(x - 12, y - 12, ad, 19.5, "#222", kalin=True))
    s.append(_yazi(w / 2, 22, "Kırmızı: Öklid · Turuncu kesikli: Manhattan", 17, "#444"))
    s.append("</svg>")
    return "".join(s)


def svg_benzerlik_cetvel(s_deg):
    w, h = 720, 180
    x0, x1, y = 60, 660, 70
    s = [svg_ac(w, h), '<defs><linearGradient id="cg" x1="0" x2="1"><stop offset="0" stop-color="#E4572E"/>'
         '<stop offset="0.5" stop-color="#F2C94C"/><stop offset="1" stop-color="#1A9C9C"/></linearGradient></defs>']
    s.append(f'<rect x="{x0}" y="{y - 12}" width="{x1 - x0}" height="24" rx="12" fill="url(#cg)"/>')
    for t in (0, 0.25, 0.5, 0.75, 1):
        x = x0 + (x1 - x0) * t
        s.append(_yazi(x, y - 20, _tr(t), 16.9, "#333", kalin=True))
        s.append(_yazi(x, y + 40, _tr(1 - t), 16.9, "#333", kalin=True))
    s.append(_yazi(x0 - 10, y - 20, "s:", 18.2, LACIVERT, "end", True))
    s.append(_yazi(x0 - 10, y + 40, "d:", 18.2, MERCAN, "end", True))
    x = x0 + (x1 - x0) * s_deg
    s.append(f'<polygon points="{x - 10},{y - 34} {x + 10},{y - 34} {x},{y - 14}" fill="{LACIVERT}"/>')
    s.append(_yazi(x0, y + 70, "← hiç benzemez (s = 0, d = 1)", 16.9, MERCAN, "start", True))
    s.append(_yazi(x1, y + 70, "özdeş (s = 1, d = 0) →", 16.9, TURKUAZ, "end", True))
    s.append("</svg>")
    return "".join(s)


# ===========================================================================
# VERİ SETLERİ VE SÖZLÜK
# ===========================================================================
# Gerçek veri: Iris versicolor (Anderson 1935; Fisher 1936), 50 bitki, cm
IRIS_VERS_CANAK = [7.0, 6.4, 6.9, 5.5, 6.5, 5.7, 6.3, 4.9, 6.6, 5.2, 5.0, 5.9, 6.0, 6.1, 5.6, 6.7, 5.6, 5.8, 6.2, 5.6,
                   5.9, 6.1, 6.3, 6.1, 6.4, 6.6, 6.8, 6.7, 6.0, 5.7, 5.5, 5.5, 5.8, 6.0, 5.4, 6.0, 6.7, 6.3, 5.6, 5.5,
                   5.5, 6.1, 5.8, 5.0, 5.6, 5.7, 5.7, 6.2, 5.1, 5.7]
IRIS_VERS_TAC = [4.7, 4.5, 4.9, 4.0, 4.6, 4.5, 4.7, 3.3, 4.6, 3.9, 3.5, 4.2, 4.0, 4.7, 3.6, 4.4, 4.5, 4.1, 4.5, 3.9,
                 4.8, 4.0, 4.9, 4.7, 4.3, 4.4, 4.8, 5.0, 4.5, 3.5, 3.8, 3.7, 3.9, 5.1, 4.5, 4.5, 4.7, 4.4, 4.1, 4.0,
                 4.4, 4.6, 4.0, 3.3, 4.2, 4.2, 4.2, 4.3, 3.0, 4.1]

# Gerçek veri: Iris, her türden ilk 5 bitki (çanak boyu, çanak eni, taç boyu, taç eni; cm)
IRIS15_AD = ["se1", "se2", "se3", "se4", "se5", "ve1", "ve2", "ve3", "ve4", "ve5", "vi1", "vi2", "vi3", "vi4", "vi5"]
IRIS15 = [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2], [4.6, 3.1, 1.5, 0.2], [5.0, 3.6, 1.4, 0.2],
          [7.0, 3.2, 4.7, 1.4], [6.4, 3.2, 4.5, 1.5], [6.9, 3.1, 4.9, 1.5], [5.5, 2.3, 4.0, 1.3], [6.5, 2.8, 4.6, 1.5],
          [6.3, 3.3, 6.0, 2.5], [5.8, 2.7, 5.1, 1.9], [7.1, 3.0, 5.9, 2.1], [6.3, 2.9, 5.6, 1.8], [6.5, 3.0, 5.8, 2.2]]
IRIS_DEG = ["Çanak boyu", "Çanak eni", "Taç boyu", "Taç eni"]

# Öğretici (kurgusal) veriler -------------------------------------------------
ISSR_AD = ["A1", "A2", "A3", "A4", "A5", "A6"]
ISSR = [  # 12 bant (1 = var, 0 = yok)
    [1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
    [1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
    [1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1],
    [0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1],
]

MORF_AD = ["M1", "M2", "M3", "M4", "M5", "M6"]
MORF_DEG = ["Tane ağırlığı (g)", "SÇKM (%)", "Salkım ağırlığı (g)", "Tane eni (mm)"]
MORF = [
    [2.2, 23.5, 310, 14.5],
    [2.5, 22.8, 640, 15.0],
    [6.1, 17.0, 330, 21.5],
    [5.8, 16.5, 650, 21.0],
    [4.0, 20.0, 470, 18.0],
    [2.4, 23.0, 480, 14.8],
]

EKS_AD = ["Gen1", "Gen2", "Gen3", "Gen4", "Gen5", "Gen6"]
EKS_ZAMAN = ["0 sa", "2 sa", "6 sa", "12 sa", "24 sa", "48 sa"]
EKS = [  # log2 ekspresyon
    [2.0, 3.0, 4.0, 5.0, 6.0, 6.5],
    [8.0, 9.0, 10.0, 11.0, 12.0, 12.5],
    [6.5, 6.0, 5.0, 4.0, 3.0, 2.0],
    [12.5, 12.0, 11.0, 10.0, 9.0, 8.0],
    [7.0, 7.2, 6.8, 7.1, 6.9, 7.0],
    [3.0, 6.0, 3.2, 6.1, 3.1, 6.0],
]

TUR_AD = ["Alan 1", "Alan 2", "Alan 3", "Alan 4", "Alan 5"]
TURLER = ["Tür a", "Tür b", "Tür c", "Tür d", "Tür e", "Tür f", "Tür g", "Tür h", "Tür i", "Tür j"]
TUR_BOLLUK = [  # birey sayısı
    [12, 5, 0, 0, 3, 0, 0, 0, 0, 0],
    [10, 6, 1, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 8, 14, 0, 2, 0, 0, 0, 0],
    [0, 1, 7, 12, 0, 3, 0, 0, 0, 0],
    [2, 0, 0, 0, 0, 0, 9, 4, 6, 1],
]

VERI_SETLERI = {
    "ISSR markör matrisi (var/yok)": dict(ad=ISSR_AD, X=ISSR, deg=[f"B{i}" for i in range(1, 13)], tur="ikili",
                                          oneri="Jaccard uzaklığı (1 − J)", kurgu=True),
    "Morfolojik ölçümler (farklı birimler)": dict(ad=MORF_AD, X=MORF, deg=MORF_DEG, tur="nicel",
                                                  oneri="Öklid", kurgu=True, standart=True),
    "Gen ekspresyon profili (log2)": dict(ad=EKS_AD, X=EKS, deg=EKS_ZAMAN, tur="nicel",
                                          oneri="Korelasyon uzaklığı (1 − r)", kurgu=True),
    "Tür bolluğu (birey sayısı)": dict(ad=TUR_AD, X=TUR_BOLLUK, deg=TURLER, tur="sayim",
                                       oneri="Bray–Curtis", kurgu=True),
    "Iris çiçekleri (gerçek veri, 15 bitki)": dict(ad=IRIS15_AD, X=IRIS15, deg=IRIS_DEG, tur="nicel",
                                                   oneri="Öklid", kurgu=False),
}

VERI_OLCU = [  # (veri türü, uygun ölçü, neden)
    ("Markör bantları (ISSR, RAPD, AFLP: baskın)", "Jaccard ya da Dice",
     "Ortak bant yokluğu (0-0) iki bireyin benzerliğine kanıt değildir; bant yokluğunun birçok nedeni olabilir."),
    ("Kodominant markör (SSR) allel verisi", "Allel paylaşımı / Nei uzaklığı",
     "Alleller ve heterozigotluk bilgisi kullanılır."),
    ("Morfolojik nicel ölçümler (farklı birimler)", "Standartlaştırılmış Öklid (z-skoru) ya da Mahalanobis",
     "Birimi büyük değişken (g, mm) Öklid'i tek başına belirler; önce standartlaştırılır. Değişkenler ilişkiliyse Mahalanobis."),
    ("Karışık veri (nicel + nitel + ikili)", "Gower katsayısı",
     "Her karakteri kendi türüne göre 0–1 aralığında karşılaştırıp ortalamasını alır."),
    ("Gen ekspresyon profilleri", "Korelasyon uzaklığı (1 − r)",
     "Genlerin düzeyinden çok inip çıkma örüntüsü (birlikte düzenlenme) önemlidir."),
    ("Tür var/yok listeleri (ekoloji)", "Jaccard ya da Sørensen (= Dice)",
     "İki alanda birlikte bulunmayan türler alanların benzerliğini göstermez."),
    ("Tür bolluk (sayım) verisi", "Bray–Curtis",
     "Ortak sıfırları yok sayar, bolluk farklarını ölçeklenmiş olarak değerlendirir."),
    ("DNA / protein dizileri", "p-uzaklığı, Jukes–Cantor, Kimura",
     "Farklı nükleotit oranı; çoklu yer değişimleri için düzeltme yapılır."),
]

ONEMLI_KISILER = [
    ("Michel Adanson (1763)", "Bütün karakterlere eşit ağırlık verme fikri: Adanson ilkesi."),
    ("Paul Jaccard (1901)", "Alpler'deki bitki floralarını karşılaştırmak için Jaccard katsayısı."),
    ("P. C. Mahalanobis (1936)", "Değişkenler arasındaki korelasyonu hesaba katan uzaklık."),
    ("Sokal ve Michener (1958)", "UPGMA ve basit eşleşme katsayısı."),
    ("Sokal ve Rohlf (1962)", "Kofenetik korelasyon: dendrogramın özgün uzaklıklara uyumunu ölçme."),
    ("Sokal ve Sneath (1963)", "<i>Principles of Numerical Taxonomy</i>: numerik taksonominin kurucu kitabı."),
    ("Sneath ve Sokal (1973)", "<i>Numerical Taxonomy</i>: yöntemlerin genişletilmiş hâli."),
    ("Saitou ve Nei (1987)", "Neighbor-Joining (komşu birleştirme) yöntemi."),
]

OTU_OYUN = [  # (öğe, rol)
    ("Çalışmaya alınan 20 asma çeşidinin her biri", "OTU"),
    ("Tane rengi (siyah / beyaz)", "Karakter"),
    ("Siyah", "Karakter durumu"),
    ("Yaprak lob sayısı", "Karakter"),
    ("5 lob", "Karakter durumu"),
    ("ISSR-7 primerinin 650 bç'lik bandı", "Karakter"),
    ("Bant var (1)", "Karakter durumu"),
    ("İncelenen 12 zeytin popülasyonu", "OTU"),
    ("Herbaryumdaki her örnek", "OTU"),
    ("Taç yaprağı boyu", "Karakter"),
    ("3,4 cm", "Karakter durumu"),
    ("Tüylülük (yok / seyrek / yoğun)", "Karakter"),
]

KARAKTER_TIP_OYUN = [  # (karakter, tip)
    ("Bant var / yok", "İkili (iki durumlu)"),
    ("Çiçek tipi: erkek / dişi / erselik", "Çok durumlu nitel (sırasız)"),
    ("Tüylülük: yok < seyrek < yoğun", "Sıralı (ordinal)"),
    ("Tane ağırlığı (g)", "Nicel sürekli"),
    ("Yaprak lob sayısı", "Nicel kesikli"),
    ("Tane rengi: beyaz / pembe / kırmızı / siyah", "Çok durumlu nitel (sırasız)"),
    ("Hastalık skoru 1–5", "Sıralı (ordinal)"),
    ("Dikenlilik: var / yok", "İkili (iki durumlu)"),
    ("SÇKM (%)", "Nicel sürekli"),
    ("Çekirdek sayısı", "Nicel kesikli"),
]

YAKIN_OYUN = [  # (ifade, yorum)
    ("Jaccard benzerliği = 0,95", "Çok benzer"),
    ("Jaccard benzerliği = 0,08", "Çok farklı"),
    ("Jaccard uzaklığı = 0,05", "Çok benzer"),
    ("Jaccard uzaklığı = 0,92", "Çok farklı"),
    ("Basit eşleşme katsayısı = 1", "Özdeş"),
    ("Öklid uzaklığı = 0", "Özdeş"),
    ("Korelasyon r = 0,98 (ekspresyon profili)", "Çok benzer"),
    ("Korelasyon uzaklığı 1 − r = 1,9", "Zıt örüntü"),
    ("Korelasyon r = −0,97", "Zıt örüntü"),
    ("Dice benzerliği = 0,1", "Çok farklı"),
    ("Bray–Curtis = 0", "Özdeş"),
    ("Bray–Curtis = 1", "Hiç ortak tür yok"),
]

TERIMLER = [
    ("Kümeleme analizi", "Nesneleri, aralarındaki benzerliğe göre, kendi içinde benzer ve birbirinden farklı gruplara ayıran çok değişkenli yöntemler."),
    ("Numerik taksonomi", "Organizmaları çok sayıda karakterin sayısal olarak karşılaştırılmasıyla, benzerliklerine göre sınıflandırma (Sokal ve Sneath, 1963)."),
    ("Fenetik", "Sınıflandırmayı evrimsel soy ilişkisine değil, genel benzerliğe dayandıran yaklaşım."),
    ("OTU", "Operasyonel Taksonomik Birim: sınıflandırılan birim (birey, çeşit, popülasyon, tür)."),
    ("Karakter (attribute)", "OTU'ları karşılaştırmak için kullanılan özellik ya da değişken (ör. tane rengi, bant 7)."),
    ("Karakter durumu", "Bir karakterin bir OTU'da aldığı değer (ör. siyah, 1, 3,4 cm)."),
    ("Veri matrisi", "Satırlarda OTU'lar, sütunlarda karakterler bulunan tablo (n × t)."),
    ("Adanson ilkesi", "Önceden bir karakteri ötekinden önemli saymadan, çok sayıda karaktere eşit ağırlık verme."),
    ("Benzerlik (yakınlık)", "İki OTU'nun ne kadar benzediği; genellikle 0 (hiç benzemez) ile 1 (özdeş) arasında."),
    ("Uzaklık (farklılık)", "İki OTU'nun ne kadar farklı olduğu; 0 özdeş demektir, büyüdükçe farklılık artar."),
    ("Metrik", "Negatif olmama, özdeşlik, simetri ve üçgen eşitsizliğini sağlayan uzaklık ölçüsü."),
    ("Üçgen eşitsizliği", "d(A,C) ≤ d(A,B) + d(B,C): doğrudan yol dolambaçlı yoldan uzun olamaz."),
    ("Öklid uzaklığı", "Karelerin toplamının karekökü: √Σ(xᵢ − yᵢ)²; düz çizgi uzaklığı."),
    ("Manhattan uzaklığı", "Mutlak farkların toplamı: Σ|xᵢ − yᵢ|; şehir bloğu uzaklığı."),
    ("Standartlaştırma", "Değişkenleri ortak ölçeğe getirme; z = (x − ortalama) / standart sapma."),
    ("Mahalanobis uzaklığı", "Değişkenlerin varyans ve korelasyonunu hesaba katan uzaklık: √[(x − y)ᵀ S⁻¹ (x − y)]."),
    ("Kovaryans matrisi", "Değişkenlerin varyanslarını (köşegen) ve kovaryanslarını (köşegen dışı) içeren matris."),
    ("Jaccard katsayısı", "a / (a + b + c): ortak sıfırları (d) dikkate almayan ikili benzerlik."),
    ("Basit eşleşme katsayısı", "(a + d) / (a + b + c + d): ortak varlık ve ortak yokluğu birlikte sayar."),
    ("Dice (Sørensen) katsayısı", "2a / (2a + b + c): ortak varlıklara iki kat ağırlık veren benzerlik."),
    ("Ortak sıfır (negatif eşleşme)", "İki OTU'da da karakterin yok (0) olması; d hücresi."),
    ("Uzaklık matrisi", "Bütün OTU çiftleri arasındaki uzaklıkların simetrik, köşegeni sıfır tablosu."),
    ("Aglomeratif kümeleme", "Her OTU ayrı küme olarak başlar; en yakın iki küme adım adım birleştirilir."),
    ("UPGMA", "Aritmetik ortalamalı ağırlıksız çift grup yöntemi: kümeler arası uzaklık, bütün çiftlerin ortalamasıdır."),
    ("Tek bağlantı", "Kümeler arası uzaklık en yakın iki üye arasındaki uzaklıktır (en yakın komşu)."),
    ("Tam bağlantı", "Kümeler arası uzaklık en uzak iki üye arasındaki uzaklıktır (en uzak komşu)."),
    ("Dendrogram", "Hiyerarşik kümelemenin birleşme sırasını ve düzeyini gösteren ağaç diyagramı."),
    ("Fenogram", "Genel benzerliğe göre kurulan dendrogram."),
    ("Kofenetik uzaklık", "İki OTU'nun dendrogramda ilk kez birleştiği düzey."),
    ("Kofenetik korelasyon", "Özgün uzaklıklarla kofenetik uzaklıklar arasındaki korelasyon; ağacın veriyi ne kadar iyi yansıttığı."),
    ("Ultrametrik", "Bütün yaprakların kökten eşit uzaklıkta olduğu ağaç; UPGMA böyle ağaç üretir."),
    ("Neighbor-Joining", "Saitou ve Nei (1987): Q ölçütüyle komşu çiftleri bularak köksüz, toplamsal ağaç kuran yöntem."),
    ("Q matrisi", "NJ'de Q(i,j) = (n − 2)·d(i,j) − rᵢ − rⱼ; en küçük Q'lu çift komşu seçilir."),
    ("Toplamsal (additive) ağaç", "Ağaç üzerindeki dal uzunlukları toplamının özgün uzaklıklara eşit olduğu ağaç."),
    ("Köksüz ağaç", "Ortak atanın yeri belirtilmemiş ağaç; NJ köksüz ağaç verir."),
    ("Gower katsayısı", "Karışık veri türleri için her karakteri 0–1 aralığına getirip ortalayan benzerlik."),
    ("Bray–Curtis", "Σ|xᵢ − yᵢ| / Σ(xᵢ + yᵢ): bolluk verisi için farklılık ölçüsü."),
    ("Korelasyon uzaklığı", "1 − r: inip çıkma örüntüsü benzer profiller yakın çıkar."),
]


# ===========================================================================
# YARDIMCILAR
# ===========================================================================
def mat_html(M, satir, sutun=None, vurgu=None, yeni=None, od=3, kosegen=True):
    """Matris HTML tablosu. vurgu: (i, j) en küçük hücre; yeni: vurgulanacak satır/sütun dizini."""
    sutun = sutun or satir
    M = np.asarray(M, float)
    h = ['<div class="mat-kap"><table class="mat"><tr><th></th>'] + [f"<th>{c}</th>" for c in sutun] + ["</tr>"]
    for i, r in enumerate(satir):
        h.append(f"<tr><th>{r}</th>")
        for j in range(len(sutun)):
            sinif = ""
            if vurgu and (i, j) in (vurgu, vurgu[::-1]):
                sinif = "en"
            elif yeni is not None and (i == yeni or j == yeni) and i != j:
                sinif = "yeni"
            elif kosegen and i == j and len(satir) == len(sutun):
                sinif = "kosegen"
            v = M[i, j]
            metin = "–" if (kosegen and i == j and len(satir) == len(sutun)) else _tr(float(v), od)
            h.append(f'<td class="{sinif}">{metin}</td>')
        h.append("</tr>")
    h.append("</table></div>")
    st.markdown("".join(h), unsafe_allow_html=True)


def adim_kutu(no, metin):
    st.markdown(f'<div class="adim"><b class="no">{no}</b> {metin}</div>', unsafe_allow_html=True)


def formul(metin):
    st.markdown(f'<div class="formul">{metin}</div>', unsafe_allow_html=True)


def _mini_oyun(anahtar, liste, secenekler, soru):
    k = f"oyun_{anahtar}"
    if k not in st.session_state:
        st.session_state[k] = int(rng().integers(0, len(liste)))
    oge, dogru = liste[st.session_state[k]]
    st.markdown(f"**{soru}:** {oge}")
    sec = st.radio(soru, secenekler, index=None, horizontal=True, key=f"{k}_sec_{st.session_state[k]}",
                   label_visibility="collapsed")
    if sec is not None:
        if sec == dogru:
            st.success(f"✅ Doğru: {dogru}")
        else:
            st.error(f"❌ Doğrusu: {dogru}")
    if st.button("🔄 Başka örnek", key=f"{k}_yeni"):
        st.session_state[k] = int(rng().integers(0, len(liste)))
        st.rerun()


ORNEK5_AD = ["A", "B", "C", "D", "E"]
ORNEK5 = [[0, 2, 6, 10, 9],
          [2, 0, 5, 9, 8],
          [6, 5, 0, 4, 5],
          [10, 9, 4, 0, 3],
          [9, 8, 5, 3, 0]]
NJ_ORNEK_AD = ["a", "b", "c", "d", "e"]
NJ_ORNEK = [[0, 5, 9, 9, 8],
            [5, 0, 10, 10, 9],
            [9, 10, 0, 8, 7],
            [9, 10, 8, 0, 3],
            [8, 9, 7, 3, 0]]


# ===========================================================================
# SAYFALAR
# ===========================================================================
def sayfa_giris():
    svg_goster(svg_hero())
    st.markdown('<div class="hero-alt">Yüksek lisans · Biyolojik Bilimlerde Kümeleme Analizi</div>', unsafe_allow_html=True)
    st.markdown("""
Biyolojide sürekli **gruplama** yaparız: çeşitleri, popülasyonları, türleri, genleri, örnekleri. **Kümeleme analizi**,
nesneleri aralarındaki benzerliğe göre, kendi içinde benzer ve birbirinden farklı gruplara ayıran yöntemlerin genel adıdır.
Bu uygulamada önce **numerik taksonominin** mantığını kuruyoruz (OTU, karakter, veri matrisi), sonra **benzerlik ve uzaklık**
ölçülerini (Öklid, Mahalanobis, Jaccard, basit eşleşme, Dice) elle hesaplıyoruz ve uzaklık matrisinden **UPGMA** ve
**Neighbor-Joining** ile adım adım ağaç kuruyoruz. Her bölümde çözümlü örnekler ve 10 / 20 / 30 soruluk alıştırmalar var.
""")
    kartlar([("1 · Numerik taksonomi", "OTU, karakter (attribute), karakter durumu, veri matrisi, Adanson ilkesi"),
             ("2 · Uzaklık ve yakınlık", "0'a ve 1'e yakın değerler ne demek, s ↔ d dönüşümü, metrik koşulları"),
             ("3 · Öklid uzaklığı", "Elle hesap, Manhattan ile karşılaştırma, ölçeklendirmenin etkisi"),
             ("4 · Mahalanobis uzaklığı", "Kovaryans matrisi, korelasyonlu verilerde uzaklık"),
             ("5 · İkili veri", "Jaccard, basit eşleşme, Dice; ortak sıfır sorunu"),
             ("6 · UPGMA ve dendrogram", "Uzaklık matrisinden adım adım ağaç, tek/tam bağlantı, kofenetik korelasyon"),
             ("7 · Neighbor-Joining", "Q matrisi, dal uzunlukları, köksüz ağaç"),
             ("8 · Biyolojik veriler", "Markör, morfoloji, ekspresyon, tür verisi: hangi veriye hangi ölçü"),
             ("9 · Çözümlü örnekler", "Her tıklamada yeni sayılarla tam çözümlü problem"),
             ("10 · Genel tekrar", "Hafıza kartları, sözlük, karışık sorular")])
    ipucu("Sol menüden ya da yukarıdaki <b>📚 Bölüm</b> listesinden bölüm seçin. Kenar çubuğundaki <b>tohum</b> sayısını "
          "sınıfta herkes aynı girerse aynı soruları görür.")


# ---------------------------------------------------------------------------
def sayfa_nt():
    st.title("1 · Numerik taksonomi ve veri matrisi")
    kutu("<b>Numerik taksonomi</b>, organizmaları <b>çok sayıda karakterin</b> sayısal olarak karşılaştırılmasıyla, "
         "<b>genel benzerliklerine</b> göre gruplayan yaklaşımdır. Temeli Sokal ve Sneath'in <i>Principles of Numerical "
         "Taxonomy</i> (1963) kitabıyla atıldı; bugün kullandığımız kümeleme yöntemlerinin çoğu (UPGMA, Jaccard ve basit "
         "eşleşme katsayıları) bu okuldan gelir.")
    st.header("OTU, karakter ve karakter durumu")
    svg_goster(svg_veri_matrisi())
    kartlar([("OTU", "Operasyonel Taksonomik Birim: sınıflandırdığımız birim. Birey, çeşit, klon, popülasyon, tür olabilir. "
                     "'Operasyonel' denir; çünkü çalışmanın amacına göre biz belirleriz."),
             ("Karakter (attribute)", "OTU'ları karşılaştırdığımız özellik. İngilizce kaynaklarda <i>character</i>, <i>attribute</i>, "
                                      "<i>variable</i> ya da <i>feature</i> diye geçer; hepsi aynı şeydir: veri matrisinin bir sütunu."),
             ("Karakter durumu", "Karakterin bir OTU'da aldığı değer: 'siyah', '1', '3,4 cm'. Veri matrisinin bir hücresi."),
             ("Veri matrisi", "n OTU (satır) × t karakter (sütun). Bütün uzaklık ve benzerlik hesapları bu matristen yapılır.")])
    ipucu("Kısaca: <b>satır = OTU</b>, <b>sütun = karakter (attribute)</b>, <b>hücre = karakter durumu</b>. "
          "Bazı yazılımlar (ör. NTSYSpc) matrisi devrik (karakterler satırda) ister; hesap aynıdır, yalnız yön değişir.")
    st.subheader("🎮 OTU mu, karakter mi, karakter durumu mu?")
    _mini_oyun("otu", OTU_OYUN, ["OTU", "Karakter", "Karakter durumu"], "Bu öğe nedir")

    st.header("Karakter türleri ve kodlama")
    tablo(pd.DataFrame([
        ("İkili (iki durumlu)", "Bant var/yok, diken var/yok", "1 / 0", "Jaccard, basit eşleşme, Dice"),
        ("Çok durumlu nitel (sırasız)", "Çiçek tipi: erkek/dişi/erselik", "Her durum ayrı ikili sütun ya da eşleşme/eşleşmeme", "Basit eşleşme, Gower"),
        ("Sıralı (ordinal)", "Tüylülük: yok < seyrek < yoğun", "0, 1, 2", "Gower, Manhattan"),
        ("Nicel kesikli", "Yaprak lob sayısı", "Sayı olarak", "Öklid, Manhattan (standartlaştırarak)"),
        ("Nicel sürekli", "Tane ağırlığı (g), SÇKM (%)", "Ölçülen değer", "Standartlaştırılmış Öklid, Mahalanobis"),
    ], columns=["Karakter türü", "Örnek", "Kodlama", "Uygun katsayı"]))
    tip = st.radio("Bir kodlama örneği görün", ["Çok durumlu nitel", "Sıralı", "Nicel"], horizontal=True, key="nt_kod")
    if tip == "Çok durumlu nitel":
        st.markdown("**Tane rengi** (beyaz / pembe / siyah) üç ayrı ikili sütunla kodlanır:")
        tablo(pd.DataFrame([("OTU 1", "siyah", 0, 0, 1), ("OTU 2", "beyaz", 1, 0, 0), ("OTU 3", "pembe", 0, 1, 0)],
                           columns=["OTU", "Tane rengi", "beyaz", "pembe", "siyah"]))
        uyari("Sırasız durumları 1, 2, 3 diye tek sütunda kodlarsanız Öklid 'beyaz ile siyah arası, beyaz ile pembe arasının "
              "iki katı' sanır. Renkler arasında böyle bir sıra yoktur; bu yüzden ayrı sütunlar ya da eşleşme katsayısı kullanılır.")
    elif tip == "Sıralı":
        st.markdown("**Tüylülük** gerçekten sıralıdır; 0 (yok), 1 (seyrek), 2 (yoğun) kodlanır. 'Yok' ile 'yoğun' arasındaki fark "
                    "'yok' ile 'seyrek' arasındakinden büyüktür ve bu kodlama bunu korur.")
    else:
        st.markdown("**Nicel** karakterler ölçüldüğü gibi girilir; birimleri farklıysa (g, %, mm) uzaklık hesaplanmadan önce "
                    "**standartlaştırılır** (3. bölüm).")

    st.header("Numerik taksonominin ilkeleri")
    kartlar([("Çok karakter", "Sınıflandırma ne kadar çok karakterle yapılırsa o kadar sağlam olur; kaynaklarda genellikle en az birkaç düzine (50–60 ve üzeri) karakter önerilir."),
             ("Eşit ağırlık (Adanson ilkesi)", "Önceden hiçbir karaktere 'daha önemli' denmez; her karakter eşit katkı yapar. Adı Michel Adanson'dan (1763) gelir."),
             ("Genel benzerlik", "İki OTU'nun benzerliği, bütün karakterlerdeki benzerliklerinin bir fonksiyonudur."),
             ("Tekrarlanabilirlik", "Aynı veri ve yöntemle herkes aynı sonuca ulaşır; sınıflandırma kişisel yargıya daha az bağlıdır."),
             ("Fenetik sınıflandırma", "Gruplar genel benzerliğe göre kurulur; evrimsel soy ilişkisi ayrıca, filogenetik yöntemlerle çıkarılır.")])
    adim = st.slider("Çalışma akışında adım", 1, len(NT_ADIM), 3, key="nt_adim")
    svg_goster(svg_nt_akis(adim - 1))
    st.markdown(f"**{adim}. {NT_ADIM[adim - 1][0]}:** {NT_ADIM[adim - 1][1]}.")

    st.header("Fenetik ve kladistik")
    tablo(pd.DataFrame([
        ("Neye dayanır?", "Genel benzerlik (bütün karakterler)", "Ortak türemiş karakterler (sinapomorfi)"),
        ("Karakter ağırlığı", "Eşit", "Ata ve türemiş durumlar ayrılır"),
        ("Tipik yöntem", "UPGMA, tek/tam bağlantı", "Parsimoni, en büyük olabilirlik"),
        ("Ağaç", "Fenogram", "Kladogram"),
        ("Soru", "Kim kime ne kadar benziyor?", "Kim kimle daha yakın ortak atayı paylaşıyor?"),
    ], columns=["", "Fenetik (numerik taksonomi)", "Kladistik"]))
    st.caption("NJ ikisinin arasında durur: uzaklık matrisi kullanır ama evrimsel (toplamsal) ağaç kurmayı amaçlar.")
    st.subheader("🎮 Bu karakter hangi türden?")
    _mini_oyun("kartip", KARAKTER_TIP_OYUN, ["İkili (iki durumlu)", "Çok durumlu nitel (sırasız)", "Sıralı (ordinal)",
                                              "Nicel kesikli", "Nicel sürekli"], "Karakter")
    alistirma_blogu("nt", "Numerik taksonomi", gen_nt)


# ---------------------------------------------------------------------------
def sayfa_yakinlik():
    st.title("2 · Uzaklık ve yakınlık (benzerlik)")
    kutu("İki OTU'nun ilişkisini iki yoldan anlatabiliriz: <b>ne kadar benzediklerini</b> (benzerlik, yakınlık, <i>similarity</i>) "
         "ya da <b>ne kadar farklı olduklarını</b> (uzaklık, farklılık, <i>distance / dissimilarity</i>). İkisi aynı bilginin iki yüzüdür: "
         "benzerlik arttıkça uzaklık azalır.")
    c1, c2 = st.columns(2)
    c1.markdown("**Benzerlik (s)**\n- Çoğu katsayıda **0 ile 1** arasında\n- **1'e yakın → çok benzer**, 1 = özdeş\n"
                "- **0'a yakın → çok farklı**, 0 = hiç ortak özellik yok\n- Örnek: Jaccard, basit eşleşme, Dice")
    c2.markdown("**Uzaklık (d)**\n- En küçük değer **0**\n- **0'a yakın → çok benzer**, 0 = özdeş\n"
                "- **Büyüdükçe → farklılık artar**\n- Örnek: Öklid, Manhattan, Mahalanobis, 1 − Jaccard")
    uyari("En sık yapılan karışıklık: 'değer 0'a yakın' tek başına bir şey söylemez. Önce sorun: bu bir <b>benzerlik</b> mi, "
          "<b>uzaklık</b> mı? Jaccard <b>benzerliği</b> 0,05 ise OTU'lar çok farklıdır; Jaccard <b>uzaklığı</b> 0,05 ise çok benzerdir.")
    st.header("Benzerlikten uzaklığa geçiş")
    s_deg = st.slider("Benzerlik katsayısı s", 0.0, 1.0, 0.8, 0.01, key="yk_s")
    svg_goster(svg_benzerlik_cetvel(s_deg))
    k1, k2, k3 = st.columns(3)
    k1.metric("s", _tr(s_deg))
    k2.metric("d = 1 − s", _tr(1 - s_deg))
    k3.metric("d = √(1 − s)", _tr(math.sqrt(1 - s_deg), 3))
    st.markdown("0–1 aralığındaki benzerlikler için en yaygın dönüşüm **d = 1 − s**'dir. **d = √(1 − s)** dönüşümü bazı "
                "katsayılarda (ör. basit eşleşme, Jaccard) uzaklığa Öklid uzayında çizilebilme özelliği kazandırır; "
                "temel koordinat analizi (PCoA) gibi yöntemlerde tercih edilir.")
    st.header("Ölçülerin aralıkları: 0'a yakın, 1'e yakın ne demek?")
    tablo(pd.DataFrame([
        ("Jaccard, basit eşleşme, Dice", "Benzerlik", "0 – 1", "1 = özdeş", "0 = hiç ortak (pozitif) özellik yok"),
        ("1 − Jaccard, 1 − SM, 1 − Dice", "Uzaklık", "0 – 1", "0 = özdeş", "1 = tamamen farklı"),
        ("Öklid, Manhattan, Mahalanobis", "Uzaklık", "0 – ∞", "0 = özdeş", "Üst sınır yok; birime bağlı"),
        ("Pearson korelasyonu r", "Benzerlik (örüntü)", "−1 – +1", "+1 = aynı örüntü", "−1 = ters örüntü; 0 = ilişkisiz"),
        ("Korelasyon uzaklığı 1 − r", "Uzaklık", "0 – 2", "0 = aynı örüntü", "2 = tam ters örüntü"),
        ("Bray–Curtis", "Uzaklık", "0 – 1", "0 = bolluklar aynı", "1 = ortak tür yok"),
    ], columns=["Ölçü", "Tür", "Aralık", "Benzer uç", "Farklı uç"]))
    st.subheader("🎮 Ne anlama geliyor?")
    _mini_oyun("yakin", YAKIN_OYUN, ["Özdeş", "Çok benzer", "Çok farklı", "Zıt örüntü", "Hiç ortak tür yok"], "Değer")

    st.header("Bir uzaklık ölçüsünün sağlaması gereken koşullar")
    kartlar([("1 · Negatif olmama", "d(A,B) ≥ 0"), ("2 · Özdeşlik", "d(A,B) = 0 ancak A ve B özdeşse"),
             ("3 · Simetri", "d(A,B) = d(B,A)"), ("4 · Üçgen eşitsizliği", "d(A,C) ≤ d(A,B) + d(B,C)")])
    st.markdown("Dört koşulu da sağlayan ölçüye **metrik** denir. Üçgen eşitsizliğini sağlamayan ama ilk üçünü sağlayana "
                "**yarı metrik** (semimetrik) denir. UPGMA'nın ürettiği ağaçlar bunun da ötesinde **ultrametrik** koşulu sağlar: "
                "d(A,C) ≤ maks[d(A,B), d(B,C)].")
    st.subheader("🧪 Üçgen eşitsizliğini deneyin")
    c1, c2, c3 = st.columns(3)
    dab = c1.number_input("d(A,B)", 0.0, 20.0, 3.0, 0.5, key="u_ab")
    dbc = c2.number_input("d(B,C)", 0.0, 20.0, 4.0, 0.5, key="u_bc")
    dac = c3.number_input("d(A,C)", 0.0, 20.0, 6.0, 0.5, key="u_ac")
    svg_goster(svg_ucgen(dab, dbc, dac))
    if dac <= dab + dbc and dab <= dac + dbc and dbc <= dab + dac:
        st.success("Üç uzaklık da üçgen eşitsizliğini sağlıyor: bu üç değer bir düzlemde gerçek bir üçgen kurabilir.")
    else:
        st.error("Üçgen eşitsizliği bozuldu: 'doğrudan yol', dolambaçlı yoldan uzun olamaz. Bu üç değer metrik bir ölçüden gelemez.")
    tablo(pd.DataFrame([
        ("Öklid", "Metrik"), ("Manhattan", "Metrik"), ("Mahalanobis", "Metrik"),
        ("1 − Jaccard", "Metrik"), ("1 − basit eşleşme", "Metrik"),
        ("Kare Öklid", "Metrik değil (üçgen eşitsizliği bozulabilir)"), ("1 − Dice (Sørensen)", "Yarı metrik"),
        ("Bray–Curtis", "Yarı metrik"), ("1 − r (korelasyon)", "Metrik değil; √[2(1 − r)] metriktir"),
    ], columns=["Ölçü", "Durum"]))
    alistirma_blogu("yakin", "Uzaklık ve yakınlık", gen_yakin)


# ---------------------------------------------------------------------------
def sayfa_oklid():
    st.title("3 · Öklid uzaklığı")
    kutu("<b>Öklid uzaklığı</b> iki nokta arasındaki düz çizgi (kuş uçuşu) uzaklığıdır. t karakterli iki OTU için "
         "her karakterdeki farkın karesi alınır, toplanır ve karekökü alınır.")
    st.latex(r"d_{\text{Öklid}}(i,j)=\sqrt{\sum_{k=1}^{t}(x_{ik}-x_{jk})^2}\qquad d_{\text{Manhattan}}(i,j)=\sum_{k=1}^{t}|x_{ik}-x_{jk}|")
    st.header("İki karakterle görelim")
    c1, c2, c3, c4 = st.columns(4)
    ax = c1.number_input("A: x₁", 0.0, 10.0, 2.0, 0.5, key="ok_ax")
    ay = c2.number_input("A: x₂", 0.0, 10.0, 3.0, 0.5, key="ok_ay")
    bx = c3.number_input("B: x₁", 0.0, 10.0, 7.0, 0.5, key="ok_bx")
    by = c4.number_input("B: x₂", 0.0, 10.0, 6.0, 0.5, key="ok_by")
    svg_goster(svg_oklid_2d((ax, ay), (bx, by)))
    dx, dy = bx - ax, by - ay
    adim_kutu("1.", f"Farklar: Δx₁ = {_tr(bx)} − {_tr(ax)} = {_tr(dx)} ; Δx₂ = {_tr(by)} − {_tr(ay)} = {_tr(dy)}")
    adim_kutu("2.", f"Kareler: {_tr(dx)}² = {_tr(dx * dx)} ; {_tr(dy)}² = {_tr(dy * dy)}")
    adim_kutu("3.", f"Toplam: {_tr(dx * dx)} + {_tr(dy * dy)} = {_tr(dx * dx + dy * dy)} (bu <b>kare Öklid</b> uzaklığıdır)")
    adim_kutu("4.", f"Karekök: √{_tr(dx * dx + dy * dy)} = <b>{_tr(math.hypot(dx, dy), 3)}</b>")
    adim_kutu("+", f"Manhattan: |{_tr(dx)}| + |{_tr(dy)}| = <b>{_tr(abs(dx) + abs(dy))}</b> (her zaman Öklid'den büyük ya da eşit)")

    st.header("Çok karakterle: morfolojik örnek")
    st.caption("Öğretici amaçlı kurgusal veri: altı asma genotipi (M1–M6), dört karakter.")
    df = pd.DataFrame(MORF, columns=MORF_DEG)
    df.insert(0, "OTU", MORF_AD)
    tablo(df)
    c1, c2 = st.columns(2)
    i = MORF_AD.index(c1.selectbox("Birinci OTU", MORF_AD, index=0, key="ok_i"))
    j = MORF_AD.index(c2.selectbox("İkinci OTU", MORF_AD, index=2, key="ok_j"))
    x, y = np.array(MORF[i], float), np.array(MORF[j], float)
    satirlar = [(MORF_DEG[k], _tr(x[k]), _tr(y[k]), _tr(x[k] - y[k]), _tr((x[k] - y[k]) ** 2, 3)) for k in range(4)]
    tablo(pd.DataFrame(satirlar, columns=["Karakter", f"{MORF_AD[i]} (1.)", f"{MORF_AD[j]} (2.)", "Fark", "Fark²"]))
    if i == j:
        st.info("Aynı OTU'yu seçtiniz: bütün farklar 0, uzaklık 0 (özdeşlik koşulu).")
    top = float(np.sum((x - y) ** 2))
    adim_kutu("Sonuç", f"Σ fark² = {_tr(top, 3)} → d = √{_tr(top, 3)} = <b>{_tr(math.sqrt(top), 3)}</b>")
    if i != j:
        pay = (x - y) ** 2 / top * 100 if top else np.zeros(4)
        fig = go.Figure(go.Bar(x=MORF_DEG, y=pay, marker_color=[TURKUAZ, MERCAN, ERIK, TURUNCU],
                               text=[f"%{_tr(v, 1)}" for v in pay], textposition="outside"))
        fig.update_layout(title="Her karakterin Σ fark²'ye katkısı (%)", yaxis_range=[0, 115])
        ciz(fig, 330)

    st.header("Ölçeklendirme sonucu nasıl değiştirir?")
    st.markdown("Yukarıdaki grafikte gördüğünüz gibi **salkım ağırlığı** (yüzlerce gram) farkları, tane ağırlığı ya da SÇKM "
                "farklarını ezer: Öklid uzaklığı neredeyse yalnızca salkım ağırlığına bakar. Çözüm, uzaklıktan önce "
                "karakterleri **standartlaştırmaktır**.")
    st.latex(r"z_{ik}=\frac{x_{ik}-\bar{x}_k}{s_k}\qquad\qquad r_{ik}=\frac{x_{ik}-\min_k}{\max_k-\min_k}")
    secim = st.radio("Veri", ["Ham veri", "z-skoru", "Aralık (0–1)"], horizontal=True, key="ok_olcek")
    X = np.array(MORF, float)
    Xs = X if secim == "Ham veri" else z_standart(X) if secim == "z-skoru" else aralik_standart(X)
    D = uzaklik_matrisi(Xs, "Öklid")
    st.markdown(f"**Öklid uzaklık matrisi ({secim.lower()})**")
    mat_html(D, MORF_AD, od=2)
    _, b = hiyerarsik(D, MORF_AD, "upgma")
    svg_goster(svg_dendrogram(b, MORF_AD, baslik=f"UPGMA · {secim.lower()}", eksen="Öklid uzaklığı"))
    if secim == "Ham veri":
        uyari("Ham veride M1 ile M6 (aynı küçük taneli, yüksek şekerli tip) farklı kümelere düşer, çünkü salkım ağırlıkları farklı. "
              "Ağaç, biyolojik tipten çok salkım ağırlığını yansıtıyor.")
    else:
        ipucu("Standartlaştırınca her karakter eşit söz hakkı kazanır: küçük taneli–yüksek şekerli tipler (M1, M2, M6; ara tip M5 "
              "onlara katılır) ile iri taneli–düşük şekerli tipler (M3, M4) ayrılır. Numerik taksonominin 'eşit ağırlık' ilkesi de bunu ister.")
    st.caption("Not: Birimi değiştirmek (salkımı g yerine kg yazmak) standartlaştırılmamış Öklid sonucunu değiştirir; z-skoru sonucunu değiştirmez.")
    alistirma_blogu("oklid", "Öklid uzaklığı", gen_oklid)


# ---------------------------------------------------------------------------
def _elips(ort, S, k):
    deg, vek = np.linalg.eigh(S)
    t = np.linspace(0, 2 * math.pi, 120)
    daire = np.vstack([np.cos(t), np.sin(t)])
    e = vek @ np.diag(np.sqrt(deg) * k) @ daire
    return e[0] + ort[0], e[1] + ort[1]


def sayfa_mahalanobis():
    st.title("4 · Mahalanobis uzaklığı")
    kutu("<b>Mahalanobis uzaklığı</b> (P. C. Mahalanobis, 1936), değişkenlerin <b>varyanslarını</b> ve aralarındaki <b>korelasyonu</b> "
         "hesaba katar. Değişkenler birbiriyle ilişkiliyse, bu ilişkinin yönünde uzanmak 'olağandır'; ilişkiye ters yönde aynı "
         "miktarda uzaklaşmak ise çok daha 'uzak' sayılır.")
    st.latex(r"d_M(\mathbf{x},\mathbf{y})=\sqrt{(\mathbf{x}-\mathbf{y})^{\mathsf T}\,\mathbf{S}^{-1}\,(\mathbf{x}-\mathbf{y})}")
    kartlar([("S = birim matris", "Varyanslar 1, korelasyon yok → Mahalanobis = Öklid"),
             ("S köşegen", "Korelasyon yok, varyanslar farklı → standartlaştırılmış Öklid"),
             ("S tam", "Varyans + korelasyon birlikte → Mahalanobis")])
    st.header("🧪 Aynı Öklid uzaklığı, farklı Mahalanobis uzaklığı")
    c1, c2 = st.columns(2)
    rho = c1.slider("İki karakter arasındaki korelasyon ρ", -0.95, 0.95, 0.8, 0.05, key="mh_r")
    uz = c2.slider("A ve B'nin merkeze Öklid uzaklığı", 0.5, 3.0, 1.5, 0.1, key="mh_u")
    S = np.array([[1.0, rho], [rho, 1.0]])
    r = np.random.default_rng(3)
    bulut = r.multivariate_normal([0, 0], S, 300)
    yon = 1 if rho >= 0 else -1
    A = np.array([uz / math.sqrt(2), yon * uz / math.sqrt(2)])
    B = np.array([uz / math.sqrt(2), -yon * uz / math.sqrt(2)])
    dA, dB = mahalanobis(A, [0, 0], S), mahalanobis(B, [0, 0], S)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bulut[:, 0], y=bulut[:, 1], mode="markers", name="Popülasyon",
                             marker=dict(color="#9fb3c8", size=5)))
    for k, ad in ((1, "d_M = 1"), (2, "d_M = 2"), (3, "d_M = 3")):
        ex, ey = _elips([0, 0], S, k)
        fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", name=ad, line=dict(color=TURKUAZ, dash="dot" if k > 1 else "solid")))
    t = np.linspace(0, 2 * math.pi, 100)
    fig.add_trace(go.Scatter(x=uz * np.cos(t), y=uz * np.sin(t), mode="lines", name="Eşit Öklid çemberi",
                             line=dict(color="#999", dash="dash")))
    fig.add_trace(go.Scatter(x=[A[0]], y=[A[1]], mode="markers+text", text=["A"], textposition="top center", name="A",
                             marker=dict(color=YESIL, size=14)))
    fig.add_trace(go.Scatter(x=[B[0]], y=[B[1]], mode="markers+text", text=["B"], textposition="top center", name="B",
                             marker=dict(color=MERCAN, size=14)))
    fig.update_layout(title="Merkezden uzaklık", xaxis_title="Karakter 1 (standart)", yaxis_title="Karakter 2 (standart)",
                      xaxis=dict(range=[-4, 4]), yaxis=dict(range=[-4, 4], scaleanchor="x"))
    ciz(fig, 480)
    k1, k2, k3 = st.columns(3)
    k1.metric("Öklid (A ve B)", _tr(uz, 2))
    k2.metric("Mahalanobis A", _tr(dA, 2))
    k3.metric("Mahalanobis B", _tr(dB, 2))
    st.markdown(f"A ve B merkeze **aynı Öklid uzaklığında** ({_tr(uz)}). A korelasyonun yönünde durduğu için Mahalanobis'e göre "
                f"**{_tr(dA)}** uzaklıkta; B korelasyona ters yönde olduğu için **{_tr(dB)}** uzaklıkta. "
                + ("Korelasyon 0'a yaklaştıkça iki değer eşitlenir." if abs(rho) > 0.1 else "Korelasyon yok denecek kadar az: iki uzaklık neredeyse eşit."))

    st.header("Adım adım hesap (iki karakter)")
    st.markdown("Gerçek veri: **Iris versicolor**, 50 çiçek (Anderson 1935; Fisher 1936). Karakterler: çanak yaprak boyu ve taç yaprak boyu (cm).")
    X = np.column_stack([IRIS_VERS_CANAK, IRIS_VERS_TAC])
    ort = X.mean(axis=0)
    Sv = np.cov(X.T, ddof=1)
    c1, c2 = st.columns(2)
    p = np.array([c1.number_input("Çiçek P: çanak boyu", 4.0, 8.0, 6.5, 0.1, key="mh_p1"),
                  c1.number_input("Çiçek P: taç boyu", 2.5, 6.0, 4.6, 0.1, key="mh_p2")])
    q = np.array([c2.number_input("Çiçek Q: çanak boyu", 4.0, 8.0, 5.5, 0.1, key="mh_q1"),
                  c2.number_input("Çiçek Q: taç boyu", 2.5, 6.0, 4.6, 0.1, key="mh_q2")])
    f = p - q
    det = Sv[0, 0] * Sv[1, 1] - Sv[0, 1] ** 2
    Sinv = np.array([[Sv[1, 1], -Sv[0, 1]], [-Sv[0, 1], Sv[0, 0]]]) / det
    ara = Sinv @ f
    d2 = float(f @ ara)
    rr = Sv[0, 1] / math.sqrt(Sv[0, 0] * Sv[1, 1])
    adim_kutu("1.", f"Ortalama: çanak {_tr(ort[0], 3)} cm, taç {_tr(ort[1], 3)} cm. Korelasyon r = {_tr(rr, 3)}")
    adim_kutu("2.", f"Kovaryans matrisi S = [[{_tr(Sv[0, 0], 4)}, {_tr(Sv[0, 1], 4)}], [{_tr(Sv[0, 1], 4)}, {_tr(Sv[1, 1], 4)}]]")
    adim_kutu("3.", f"det(S) = {_tr(Sv[0, 0], 4)} × {_tr(Sv[1, 1], 4)} − {_tr(Sv[0, 1], 4)}² = {_tr(det, 5)}")
    adim_kutu("4.", f"S⁻¹ = (1 / det) × [[s₂₂, −s₁₂], [−s₁₂, s₁₁]] = [[{_tr(Sinv[0, 0], 3)}, {_tr(Sinv[0, 1], 3)}], "
                    f"[{_tr(Sinv[1, 0], 3)}, {_tr(Sinv[1, 1], 3)}]]")
    adim_kutu("5.", f"Fark vektörü x − y = ({_tr(f[0], 2)}, {_tr(f[1], 2)})")
    adim_kutu("6.", f"S⁻¹(x − y) = ({_tr(ara[0], 3)}, {_tr(ara[1], 3)}); (x − y)ᵀ S⁻¹ (x − y) = "
                    f"{_tr(f[0], 2)}×{_tr(ara[0], 3)} + {_tr(f[1], 2)}×{_tr(ara[1], 3)} = {_tr(d2, 3)}")
    adim_kutu("7.", f"d_M = √{_tr(d2, 3)} = <b>{_tr(math.sqrt(max(d2, 0)), 3)}</b> · Öklid = <b>{_tr(float(np.hypot(*f)), 3)}</b>")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", name="I. versicolor", marker=dict(color="#9fb3c8", size=7)))
    for k in (1, 2):
        ex, ey = _elips(ort, Sv, k)
        fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", name=f"d_M = {k}", line=dict(color=TURKUAZ, dash="dot" if k > 1 else "solid")))
    fig.add_trace(go.Scatter(x=[p[0], q[0]], y=[p[1], q[1]], mode="markers+text+lines", text=["P", "Q"], textposition="top center",
                             name="P–Q", marker=dict(color=MERCAN, size=13), line=dict(color=MERCAN)))
    fig.update_layout(title="Iris versicolor: çanak ve taç yaprak boyu", xaxis_title="Çanak boyu (cm)", yaxis_title="Taç boyu (cm)")
    ciz(fig, 440)
    ipucu("Mahalanobis'in kullanıldığı yerler: aykırı gözlem (outlier) bulma, diskriminant analizi, popülasyonlar arası "
          "morfometrik uzaklık. Dikkat: S'nin tersinin alınabilmesi için gözlem sayısı karakter sayısından fazla olmalı ve "
          "karakterler birbirinin tam tekrarı olmamalıdır; aykırı değerler S'yi bozar.")
    alistirma_blogu("mahal", "Mahalanobis uzaklığı", gen_mahal)


# ---------------------------------------------------------------------------
def _ikili_oku(metin, uzunluk=None):
    t = "".join(ch for ch in metin if ch in "01")
    return [int(ch) for ch in t]


def sayfa_ikili():
    st.title("5 · İkili veri: Jaccard, basit eşleşme ve Dice")
    kutu("Markör bantları (ISSR, RAPD, AFLP), bir türün bir alanda bulunup bulunmaması ya da bir karakterin var/yok olması "
         "<b>ikili (0/1)</b> veridir. İki OTU karşılaştırılınca her karakter dört hücreden birine düşer:")
    st.header("a, b, c, d tablosu")
    c1, c2 = st.columns(2)
    m1 = c1.text_input("OTU i (0 ve 1'lerden oluşan dizi)", "110110010100", key="ik_1")
    m2 = c2.text_input("OTU j", "110110011100", key="ik_2")
    x, y = _ikili_oku(m1), _ikili_oku(m2)
    if len(x) != len(y) or not x:
        st.error("İki dizi aynı uzunlukta olmalı ve yalnız 0 / 1 içermeli.")
        return alistirma_blogu("ikili", "İkili veri katsayıları", gen_ikili)
    a, b, c, d = abcd(x, y)
    svg_goster(svg_abcd(a, b, c, d))
    n = a + b + c + d
    J = a / (a + b + c) if a + b + c else 1.0
    SM = (a + d) / n
    Dc = 2 * a / (2 * a + b + c) if a + b + c else 1.0
    st.latex(r"J=\frac{a}{a+b+c}\qquad SM=\frac{a+d}{a+b+c+d}\qquad D=\frac{2a}{2a+b+c}")
    k1, k2, k3 = st.columns(3)
    k1.metric("Jaccard", _tr(J, 3), help=f"{a} / ({a} + {b} + {c})")
    k2.metric("Basit eşleşme", _tr(SM, 3), help=f"({a} + {d}) / {n}")
    k3.metric("Dice", _tr(Dc, 3), help=f"2×{a} / (2×{a} + {b} + {c})")
    formul(f"J = {a} / ({a} + {b} + {c}) = {_tr(J, 3)}  ·  SM = ({a} + {d}) / {n} = {_tr(SM, 3)}  ·  "
           f"D = 2×{a} / (2×{a} + {b} + {c}) = {_tr(Dc, 3)}")
    st.markdown("Uzaklık istenirse: **1 − J**, **1 − SM**, **1 − D**. Dice, ortak varlığa (a) iki kat ağırlık verir; bu yüzden "
                "her zaman Jaccard'a eşit ya da büyüktür. İkisi arasında D = 2J / (1 + J) ilişkisi vardır; sıralamaları aynıdır.")

    st.header("Ortak sıfır sorunu: hangisini seçmeli?")
    st.markdown("Fark, **d** hücresindedir (iki OTU'da da 0). Basit eşleşme ortak yokluğu benzerlik sayar; Jaccard ve Dice saymaz.")
    ek = st.slider("Diziye eklenen ortak 0-0 karakter sayısı", 0, 40, 0, key="ik_ek")
    seri = [(k, a / (a + b + c) if a + b + c else 1.0, (a + d + k) / (n + k)) for k in range(0, 41)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[s[0] for s in seri], y=[s[1] for s in seri], name="Jaccard", line=dict(color=TURKUAZ, width=3)))
    fig.add_trace(go.Scatter(x=[s[0] for s in seri], y=[s[2] for s in seri], name="Basit eşleşme", line=dict(color=MERCAN, width=3)))
    fig.add_vline(x=ek, line_dash="dash", line_color="#888")
    fig.update_layout(title="Ortak sıfır eklendikçe katsayılar", xaxis_title="Eklenen 0-0 karakter", yaxis_range=[0, 1.02])
    ciz(fig, 340)
    st.markdown(f"{ek} ortak sıfır eklenince: Jaccard **{_tr(seri[ek][1], 3)}** (değişmez), basit eşleşme **{_tr(seri[ek][2], 3)}**.")
    c1, c2 = st.columns(2)
    c1.markdown("**Jaccard / Dice seçin**\n- Baskın markörler (ISSR, RAPD, AFLP): bant yokluğu farklı nedenlerden "
                "(mutasyon, primer bağlanmaması, delesyon) olabilir; iki bireyde bant olmaması ortak bir özellik kanıtı değildir.\n"
                "- Tür listeleri: iki alanda da bulunmayan binlerce tür alanları benzer yapmaz.")
    c2.markdown("**Basit eşleşme seçin**\n- 0 ve 1 eşit anlamlıysa: örneğin 'yaprak tüylü / tüysüz' gibi iki durumlu, "
                "simetrik morfolojik karakterler.\n- Kodominant verinin ikili kodlanmasında, iki durum da gerçek bir gözlem olduğunda.")

    st.header("ISSR örneği: bütün çiftler")
    st.caption("Öğretici amaçlı kurgusal veri: altı genotip (A1–A6), 12 ISSR bandı.")
    df = pd.DataFrame(ISSR, columns=[f"B{i}" for i in range(1, 13)])
    df.insert(0, "OTU", ISSR_AD)
    tablo(df)
    n6 = len(ISSR)
    Jm = np.array([[jaccard(ISSR[i], ISSR[j]) for j in range(n6)] for i in range(n6)])
    SMm = np.array([[basit_eslesme(ISSR[i], ISSR[j]) for j in range(n6)] for i in range(n6)])
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Jaccard benzerlik matrisi**")
        mat_html(Jm, ISSR_AD, od=2, kosegen=False)
    with c2:
        st.markdown("**Basit eşleşme benzerlik matrisi**")
        mat_html(SMm, ISSR_AD, od=2, kosegen=False)
    st.markdown("Köşegen 1'dir (her OTU kendisiyle özdeş). Basit eşleşme hiçbir zaman Jaccard'dan küçük olmaz (d = 0 ise ikisi eşittir); örneğin A3–A5 için "
                "Jaccard 0,38, basit eşleşme 0,58. Aradaki fark ortak sıfırlardan gelir.")
    alistirma_blogu("ikili", "İkili veri katsayıları", gen_ikili)


# ---------------------------------------------------------------------------
def _veri_secimi(anahtar, ornek_ad, ornek_D):
    secim = st.radio("Uzaklık matrisi", ["Ders örneği (5 OTU)", "ISSR verisi (Jaccard uzaklığı)", "Rastgele 5 OTU"],
                     horizontal=True, key=f"{anahtar}_veri")
    if secim.startswith("Ders"):
        return list(ornek_ad), np.array(ornek_D, float)
    if secim.startswith("ISSR"):
        return list(ISSR_AD), uzaklik_matrisi(ISSR, "Jaccard uzaklığı (1 − J)")
    tk = f"{anahtar}_rast"
    if tk not in st.session_state:
        st.session_state[tk] = int(rng().integers(0, 10 ** 6))
    if st.button("🎲 Yeni rastgele matris", key=f"{anahtar}_rbut"):
        st.session_state[tk] = int(rng().integers(0, 10 ** 6))
    return list("ABCDE"), _rastgele_D(np.random.default_rng(st.session_state[tk]), 5)


def _rastgele_D(r, n):
    """Tam sayılı, yorumlanabilir rastgele uzaklık matrisi (noktalardan türetilir, metrik)."""
    while True:
        P = r.integers(0, 9, size=(n, 2))
        D = np.abs(P[:, None, :] - P[None, :, :]).sum(-1).astype(float)
        iu = D[np.triu_indices(n, 1)]
        if iu.min() > 0 and len(set(iu)) >= n:
            return D


def sayfa_upgma():
    st.title("6 · Uzaklık matrisi, UPGMA ve dendrogram")
    kutu("<b>Aglomeratif (birleştirici) hiyerarşik kümeleme</b> her OTU'yu ayrı bir küme olarak başlatır. Her adımda "
         "uzaklık matrisindeki <b>en küçük değeri</b> bulur, o iki kümeyi birleştirir ve yeni kümenin öteki kümelere "
         "uzaklığını yeniden hesaplar. n OTU için n − 1 adımda tek küme kalır. Yöntemleri ayıran tek şey, bu "
         "<b>yeniden hesaplama kuralıdır</b>.")
    kartlar([("UPGMA (ortalama bağlantı)", "Yeni küme ile K arasındaki uzaklık, bütün üye çiftlerinin ortalaması: "
                                         "(nᵢ·d(i,K) + nⱼ·d(j,K)) / (nᵢ + nⱼ). Sokal ve Michener (1958)."),
             ("Tek bağlantı (en yakın komşu)", "min[d(i,K), d(j,K)]. Zincirleme eğilimi vardır: kümeler uzun zincirler gibi büyür."),
             ("Tam bağlantı (en uzak komşu)", "maks[d(i,K), d(j,K)]. Sıkı, yuvarlak ve birbirine benzer büyüklükte kümeler verir.")])
    st.caption("UPGMA = Unweighted Pair Group Method with Arithmetic mean: aritmetik ortalamalı ağırlıksız çift grup yöntemi. "
               "'Ağırlıksız' denmesinin nedeni, her OTU'nun ortalamaya eşit katkı yapmasıdır.")
    st.header("🧮 Adım adım")
    adlar, D0 = _veri_secimi("up", ORNEK5_AD, ORNEK5)
    yontem = st.radio("Yöntem", list(YONTEM_AD), format_func=lambda k: YONTEM_AD[k], horizontal=True, key="up_y")
    adimlar, birlesme = hiyerarsik(D0, adlar, yontem)
    k = st.slider("Adım", 1, len(adimlar), 1, key=f"up_adim_{len(adimlar)}")
    a = adimlar[k - 1]
    c1 = c2 = st.container()
    with c1:
        st.markdown(f"**Adım {k}: başlangıç matrisi** (en küçük değer sarı)")
        mat_html(a["D"], a["adlar"], vurgu=(a["i"], a["j"]))
        adim_kutu("En küçük:", f"d({a['adlar'][a['i']]}, {a['adlar'][a['j']]}) = <b>{_tr(a['deger'], 3)}</b> → bu ikisi birleşir: "
                               f"<b>{a['yeni']}</b>, birleşme düzeyi {_tr(a['deger'], 3)}")
        if a["formul"]:
            st.markdown(f"**Yeni uzaklıklar ({YONTEM_AD[yontem]}):**")
            for hedef, f in a["formul"]:
                formul(f"d({a['yeni']}, {hedef}) = {f}")
        if k < len(adimlar):
            sonraki = adimlar[k]
            st.markdown("**Güncellenmiş matris** (yeni satır/sütun yeşil):")
            mat_html(sonraki["D"], sonraki["adlar"], yeni=0)
    with c2:
        svg_goster(svg_dendrogram(birlesme, adlar, adim=k, baslik=f"{YONTEM_AD[yontem]}: {k}. adım"))
    if k == len(adimlar):
        r = kofenetik_r(D0, birlesme)
        st.success(f"Ağaç tamamlandı. Kofenetik korelasyon r = {_tr(r, 3)}")

    st.header("Dendrogram nasıl okunur?")
    kartlar([("Yapraklar", "Her uç bir OTU'dur."),
             ("Düğüm ve düzey", "İki dalın birleştiği noktanın ekseni gösterdiği değer, o iki kümenin birleştiği uzaklıktır."),
             ("Kofenetik uzaklık", "İki OTU'nun ağaçta ilk kez aynı kümeye girdiği düzey. A ile C'nin uzaklığı, ağaçta onları ilk birleştiren düğümün düzeyidir."),
             ("Dal sırası önemsizdir", "Bir düğümdeki iki dal yer değiştirebilir; yan yana çizilen iki OTU'nun daha benzer olduğu sonucunu çıkarmayın. Önemli olan birleşme düzeyidir."),
             ("Kesim", "Ağacı belirli bir düzeyden yatay (burada dikey) kesmek, OTU'ları gruplara ayırır.")])
    tam_adim, tam_b = hiyerarsik(D0, adlar, yontem)
    hmax = max(b[2] for b in tam_b)
    kes = st.slider("Kesim düzeyi", 0.0, float(round(hmax * 1.05, 3)), float(round(hmax * 0.5, 3)), key=f"up_kes_{round(hmax, 3)}")
    svg_goster(svg_dendrogram(tam_b, adlar, kesim=kes, baslik="Kesim çizgisine göre kümeler"))
    et = kesim_kumeleri(tam_b, len(adlar), kes)
    gruplar = {}
    for ad_, e in zip(adlar, et):
        gruplar.setdefault(e, []).append(ad_)
    st.markdown(f"**{len(gruplar)} küme:** " + " · ".join("{" + ", ".join(g) + "}" for g in gruplar.values()))

    st.header("Üç yöntem aynı veride")
    sutun = st.tabs([YONTEM_AD[y].split(" (")[0] for y in YONTEM_AD])
    for kol, y in zip(sutun, YONTEM_AD):
        _, b = hiyerarsik(D0, adlar, y)
        with kol:
            svg_goster(svg_dendrogram(b, adlar, baslik=YONTEM_AD[y].split(" (")[0], olcek_max=float(np.max(D0))))
            st.caption(f"Kofenetik r = {_tr(kofenetik_r(D0, b), 3)}")
    st.markdown("Tek bağlantı birleşmeleri hep en küçük değerden yapar, bu yüzden düzeyleri en düşüktür; tam bağlantı en yükseğini "
                "kullanır. UPGMA ikisinin arasında kalır ve kofenetik korelasyonu genellikle en yüksek olandır; numerik taksonomide "
                "en çok kullanılan yöntem budur.")
    st.header("Kofenetik korelasyon")
    st.markdown("Ağaç, özgün uzaklıkları ne kadar iyi yansıtıyor? Her OTU çifti için **özgün uzaklık** ile **kofenetik uzaklık** "
                "(ağaçtaki birleşme düzeyi) yan yana konur ve Pearson korelasyonu hesaplanır.")
    n = len(adlar)
    C = kofenetik(tam_b, n)
    iu = np.triu_indices(n, 1)
    cift = [f"{adlar[i]}–{adlar[j]}" for i, j in zip(*iu)]
    tablo(pd.DataFrame({"Çift": cift, "Özgün d": [_tr(v, 3) for v in D0[iu]], "Kofenetik d": [_tr(v, 3) for v in C[iu]]}))
    fig = go.Figure(go.Scatter(x=D0[iu], y=C[iu], mode="markers+text", text=cift, textposition="top center",
                               marker=dict(color=TURKUAZ, size=10)))
    fig.update_layout(title=f"r = {_tr(kofenetik_r(D0, tam_b), 3)}", xaxis_title="Özgün uzaklık", yaxis_title="Kofenetik uzaklık",
                      showlegend=False)
    ciz(fig, 380)
    ipucu("Kofenetik korelasyonu Sokal ve Rohlf (1962) önerdi. Yaygın yorum ölçütü (Rohlf, NTSYSpc kılavuzu): r ≥ 0,9 çok iyi, "
          "0,8–0,9 iyi, 0,7–0,8 zayıf, 0,7'nin altı çok zayıf uyum. Markör çalışmalarında r değeri raporlanmalıdır.")
    uyari("UPGMA bütün yaprakları kökten eşit uzaklıkta çizer (<b>ultrametrik</b> ağaç). Bu, filogenide 'bütün soy hatlarında "
          "değişim hızı eşit' (moleküler saat) varsayımı demektir. Hızlar farklıysa UPGMA yanlış topoloji verebilir; o zaman NJ "
          "gibi yöntemler kullanılır. Filogenetik UPGMA ağaçlarında dal uzunluğu, birleşme düzeyinin yarısıdır (d/2).")
    alistirma_blogu("upgma", "UPGMA ve dendrogram", gen_upgma)


# ---------------------------------------------------------------------------
def sayfa_nj():
    st.title("7 · Neighbor-Joining (komşu birleştirme)")
    kutu("<b>Neighbor-Joining (NJ)</b>, Saitou ve Nei (1987) tarafından geliştirildi. UPGMA gibi uzaklık matrisinden başlar, "
         "ama en küçük uzaklığı değil, <b>Q ölçütü en küçük olan çifti</b> birleştirir. Q, bir çiftin birbirine yakınlığını, "
         "her birinin <b>öteki bütün OTU'lardan ne kadar uzak olduğuyla</b> düzeltir. Böylece hızlı evrimleşen (uzun dallı) "
         "OTU'lar yanlışlıkla bir araya getirilmez. Sonuç <b>köksüz</b> ve <b>toplamsal</b> bir ağaçtır.")
    st.latex(r"r_i=\sum_{k} d(i,k)\qquad Q(i,j)=(n-2)\,d(i,j)-r_i-r_j")
    st.latex(r"\delta_i=\frac{d(i,j)}{2}+\frac{r_i-r_j}{2(n-2)}\qquad \delta_j=d(i,j)-\delta_i\qquad d(u,k)=\frac{d(i,k)+d(j,k)-d(i,j)}{2}")
    st.caption("rᵢ: i'nin bütün öteki OTU'lara uzaklıklarının toplamı · n: o adımdaki OTU (düğüm) sayısı · δ: dal uzunluğu · u: yeni iç düğüm")
    st.header("🧮 Adım adım")
    adlar, D0 = _veri_secimi("nj", NJ_ORNEK_AD, NJ_ORNEK)
    adimlar, kenarlar = komsu_birlestirme(D0, adlar)
    k = st.slider("Adım", 1, len(adimlar), 1, key=f"nj_adim_{len(adimlar)}")
    a = adimlar[k - 1]
    if a.get("son"):
        st.markdown("**Son adım:** iki düğüm kaldı; aralarındaki uzaklık son dalın uzunluğudur.")
        mat_html(a["D"], a["adlar"])
        formul(f"Dal {a['adlar'][0]} – {a['adlar'][1]} = {_tr(float(a['D'][0, 1]), 3)}")
    else:
        n = len(a["adlar"])
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Uzaklık matrisi (n = {n})**")
            mat_html(a["D"], a["adlar"], vurgu=(a["i"], a["j"]))
            st.markdown("**Satır toplamları rᵢ:** " + " · ".join(f"r({ad_}) = {_tr(v, 3)}" for ad_, v in zip(a["adlar"], a["r"])))
        with c2:
            st.markdown(f"**Q matrisi** (en küçük Q sarı)")
            mat_html(a["Q"], a["adlar"], vurgu=(a["i"], a["j"]))
        ai, aj = a["adlar"][a["i"]], a["adlar"][a["j"]]
        dij = float(a["D"][a["i"], a["j"]])
        ri, rj = float(a["r"][a["i"]]), float(a["r"][a["j"]])
        adim_kutu("1.", f"Q({ai},{aj}) = ({n} − 2)×{_tr(dij, 3)} − {_tr(ri, 3)} − {_tr(rj, 3)} = <b>{_tr(a['Qmin'], 3)}</b> → en küçük: "
                        f"<b>{ai}</b> ile <b>{aj}</b> komşudur, yeni düğüm <b>{a['u']}</b>")
        adim_kutu("2.", f"δ({ai}) = {_tr(dij, 3)}/2 + ({_tr(ri, 3)} − {_tr(rj, 3)}) / (2×{n - 2}) = <b>{_tr(a['di'], 3)}</b> ; "
                        f"δ({aj}) = {_tr(dij, 3)} − {_tr(a['di'], 3)} = <b>{_tr(a['dj'], 3)}</b>")
        st.markdown(f"**3. {a['u']} düğümünün öteki düğümlere uzaklığı:**")
        for hedef, v in a["yeni"]:
            ik = float(a["D"][a["i"], a["adlar"].index(hedef)])
            jk = float(a["D"][a["j"], a["adlar"].index(hedef)])
            formul(f"d({a['u']}, {hedef}) = ({_tr(ik, 3)} + {_tr(jk, 3)} − {_tr(dij, 3)}) / 2 = {_tr(v, 3)}")
        if min(a["di"], a["dj"]) < 0:
            uyari("Negatif dal uzunluğu çıktı. Bu, verinin tam toplamsal olmadığını gösterir; uygulamada 0 kabul edilir ve "
                  "çizimde 0 olarak gösterilir.")
    yapraklar = list(adlar)
    gorunen = kenarlar[: 2 * k] if not a.get("son") else kenarlar
    if a.get("son"):
        svg_goster(svg_nj_agac(kenarlar, yapraklar, "Tamamlanmış NJ ağacı (köksüz)"))
    else:
        st.caption(f"Bu adıma kadar bağlanan dallar: " + " · ".join(f"{x}–{y}: {_tr(l, 3)}" for x, y, l in gorunen))
        svg_goster(svg_nj_agac(kenarlar, yapraklar, "Tamamlanmış ağaç (bu adımda birleşen çift kırmızı)", vurgu=(a["u"],)))
    st.header("Ağaç uzaklıkları veriyi karşılıyor mu?")
    st.markdown("NJ ağacında iki yaprak arasındaki **yol uzunluğu** (dalların toplamı), veri toplamsal ise özgün uzaklığa eşittir.")
    iu = np.triu_indices(len(adlar), 1)
    satir = []
    for i, j in zip(*iu):
        yol = agac_yol_uzakligi(kenarlar, adlar[i], adlar[j])
        satir.append((f"{adlar[i]}–{adlar[j]}", _tr(float(D0[i, j]), 3), _tr(yol, 3), _tr(yol - float(D0[i, j]), 3)))
    tablo(pd.DataFrame(satir, columns=["Çift", "Özgün d", "Ağaçtaki yol", "Fark"]))
    st.header("UPGMA ile NJ karşılaştırması")
    tablo(pd.DataFrame([
        ("Birleştirme ölçütü", "En küçük d(i,j)", "En küçük Q(i,j)"),
        ("Varsayım", "Sabit değişim hızı (moleküler saat), ultrametrik veri", "Toplamsal veri; hızlar farklı olabilir"),
        ("Ağaç", "Köklü, ultrametrik (yapraklar hizalı)", "Köksüz, dal uzunlukları farklı"),
        ("Dal uzunluğu", "Birleşme düzeyinin yarısı", "δ formülüyle ayrı ayrı"),
        ("Tipik kullanım", "Fenetik sınıflandırma, markör verisi", "Filogenetik uzaklık ağaçları (DNA dizileri)"),
        ("Köklendirme", "Kendiliğinden köklü", "Dış grupla ya da orta noktadan köklenir"),
    ], columns=["", "UPGMA", "Neighbor-Joining"]))
    ipucu("Ders örneğindeki (a–e) matris tam toplamsaldır: tablodaki farkların hepsi 0 çıkar. Gerçek verilerde küçük farklar olur; "
          "NJ bu farkları en aza indiren ağaca yaklaşır. Köksüz ağacı köklemek için bir <b>dış grup</b> (ör. akraba bir tür) eklenir.")
    alistirma_blogu("nj", "Neighbor-Joining", gen_nj)


# ---------------------------------------------------------------------------
def sayfa_biyoveri():
    st.title("8 · Biyolojik veriler: hangi veriye hangi ölçü?")
    kutu("Doğru ölçüyü seçmek, kümeleme yöntemini seçmek kadar önemlidir. Sorulacak soru: <b>verimin türü ne ve iki OTU'yu "
         "benzer yapan şey ne?</b>")
    tablo(pd.DataFrame(VERI_OLCU, columns=["Veri", "Uygun ölçü", "Neden"]))
    st.header("🧪 Kendi analizinizi yapın")
    c1, c2, c3 = st.columns(3)
    vs = c1.selectbox("Veri seti", list(VERI_SETLERI), key="bv_veri")
    v = VERI_SETLERI[vs]
    olculer = list(OLCULER)
    olcu = c2.selectbox("Uzaklık ölçüsü", olculer, index=olculer.index(v["oneri"]), key=f"bv_olcu_{vs}")
    yontem = c3.selectbox("Kümeleme", list(YONTEM_AD), format_func=lambda k: YONTEM_AD[k], key="bv_y")
    std = False
    if v["tur"] == "nicel":
        std = st.checkbox("Karakterleri z-skoruna göre standartlaştır", value=v.get("standart", False), key=f"bv_std_{vs}")
    if v["kurgu"]:
        st.caption("Öğretici amaçlı kurgusal veri.")
    else:
        st.caption("Gerçek veri: Iris (Anderson 1935; Fisher 1936); se = I. setosa, ve = I. versicolor, vi = I. virginica.")
    X = np.array(v["X"], float)
    df = pd.DataFrame(X, columns=v["deg"])
    df.insert(0, "OTU", v["ad"])
    with st.expander("Veri matrisi", expanded=False):
        tablo(df)
    Xk = z_standart(X) if std else X
    ikili_olcu = any(t in olcu for t in ("Jaccard", "eşleşme", "Dice"))
    if ikili_olcu and v["tur"] != "ikili":
        uyari("Bu ölçü ikili (0/1) veri içindir. Seçtiğiniz veri ikili değil; sonuç anlamlı olmaz.")
        Xk = (X > 0).astype(float)
        st.caption("Hesap için veri var (> 0) / yok olarak ikiye çevrildi.")
    D = uzaklik_matrisi(Xk, olcu)
    _, b = hiyerarsik(D, v["ad"], yontem)
    c1, c2 = st.tabs(["🌳 Dendrogram", "🟥 Uzaklık matrisi (ısı haritası)"])
    with c1:
        svg_goster(svg_dendrogram(b, v["ad"], baslik=f"{olcu} · {YONTEM_AD[yontem].split(' (')[0]}", eksen=olcu))
        st.caption(f"Kofenetik korelasyon r = {_tr(kofenetik_r(D, b), 3)}")
    with c2:
        fig = go.Figure(go.Heatmap(z=D, x=v["ad"], y=v["ad"], colorscale="Tealrose", colorbar=dict(title="d")))
        fig.update_layout(title="Uzaklık matrisi", yaxis=dict(autorange="reversed"))
        ciz(fig, 420)
    if olcu != v["oneri"]:
        st.info(f"Bu veri için genellikle önerilen ölçü: **{v['oneri']}**. Farkı görmek için ikisini karşılaştırın.")
    if v["tur"] == "nicel" and olcu in ("Öklid", "Manhattan") and not std and vs.startswith("Morf"):
        uyari("Morfolojik karakterlerin birimleri farklı; standartlaştırmadan Öklid, salkım ağırlığına göre kümeler.")

    st.header("Ekspresyon verisinde Öklid ile korelasyon uzaklığı")
    fig = go.Figure()
    for i, ad_ in enumerate(EKS_AD):
        fig.add_trace(go.Scatter(x=EKS_ZAMAN, y=EKS[i], mode="lines+markers", name=ad_, line=dict(color=KUME_RENK[i], width=3)))
    fig.update_layout(title="Altı genin zamana göre ekspresyonu (log2, kurgusal)", yaxis_title="log2 ekspresyon")
    ciz(fig, 380)
    c1, c2 = st.tabs(["Öklid", "Korelasyon uzaklığı (1 − r)"])
    for kol, o in zip((c1, c2), ("Öklid", "Korelasyon uzaklığı (1 − r)")):
        D2 = uzaklik_matrisi(EKS, o)
        _, b2 = hiyerarsik(D2, EKS_AD, "upgma")
        with kol:
            svg_goster(svg_dendrogram(b2, EKS_AD, baslik=o, eksen=o))
    st.markdown("Gen1 ile Gen2 aynı biçimde artıyor ama düzeyleri çok farklı. **Öklid** onları ayırır, çünkü düzey farkına bakar; "
                "**korelasyon uzaklığı** birleştirir, çünkü örüntü aynıdır. Birlikte düzenlenen (ko-regüle) genleri arıyorsanız "
                "korelasyon uzaklığı doğru seçimdir. Gen3 ve Gen4 ise ters örüntüdedir ve Gen1–Gen2'den en uzak gruptur.")
    st.header("Tür verisinde Jaccard ve Bray–Curtis")
    st.markdown("Bolluk verisinde (kaç birey) **Bray–Curtis**, yalnız var/yok bilgisi varsa **Jaccard** kullanılır. İkisi de "
                "iki alanda birlikte bulunmayan türleri benzerlik saymaz. Alan 5'in öteki alanlarla neredeyse hiç ortak türü yoktur; "
                "Bray–Curtis uzaklığı 0,9–1 arasında çıkar.")
    st.subheader("🎮 Hangi ölçü?")
    _mini_oyun("olcu", [(v_[0], v_[1]) for v_ in VERI_OLCU], sorted({v_[1] for v_ in VERI_OLCU}), "Veri")
    alistirma_blogu("biyo", "Biyolojik verilerde ölçü seçimi", gen_biyo)


# ---------------------------------------------------------------------------
ORNEK_TURLERI = ["Öklid ve Manhattan uzaklığı", "Benzerlikten uzaklığa", "Jaccard, basit eşleşme, Dice",
                 "Mahalanobis (iki karakter)", "UPGMA (4 OTU)", "Tek ve tam bağlantı (4 OTU)", "Neighbor-Joining (4 OTU)",
                 "Kofenetik uzaklık"]


def ornek_uret(tur, r):
    """(soru_md, [adım_md, ...], cevap_md) döndürür."""
    if tur == ORNEK_TURLERI[0]:
        t = int(r.integers(3, 6))
        x = r.integers(1, 10, t)
        y = r.integers(1, 10, t)
        fark = y - x
        soru = f"İki OTU'nun {t} karakteri: **i = ({', '.join(map(str, x))})**, **j = ({', '.join(map(str, y))})**. Öklid ve Manhattan uzaklığını bulun."
        top = int(np.sum(fark ** 2))
        adim = [f"Farklar (j − i): {', '.join(map(str, fark))}",
                f"Kareler: {', '.join(str(int(f * f)) for f in fark)} → toplam = **{top}**",
                f"Öklid = √{top} = **{_tr(math.sqrt(top), 3)}**",
                f"Mutlak farklar: {', '.join(str(abs(int(f))) for f in fark)} → Manhattan = **{int(np.sum(np.abs(fark)))}**"]
        return soru, adim, f"Öklid = {_tr(math.sqrt(top), 3)}, Manhattan = {int(np.sum(np.abs(fark)))}"
    if tur == ORNEK_TURLERI[1]:
        s = int(r.integers(5, 96)) / 100
        soru = f"İki OTU arasındaki benzerlik s = {_tr(s)}. (a) d = 1 − s, (b) d = √(1 − s) nedir? (c) OTU'lar benzer mi?"
        adim = [f"(a) d = 1 − {_tr(s)} = **{_tr(1 - s)}**", f"(b) d = √{_tr(1 - s)} = **{_tr(math.sqrt(1 - s), 3)}**",
                "(c) " + ("s 1'e yakın: OTU'lar **çok benzer**." if s >= 0.75 else "s 0'a yakın: OTU'lar **çok farklı**." if s <= 0.25
                          else "s orta düzeyde: OTU'lar **kısmen benzer**.")]
        return soru, adim, f"d = {_tr(1 - s)}; √(1 − s) = {_tr(math.sqrt(1 - s), 3)}"
    if tur == ORNEK_TURLERI[2]:
        n = int(r.integers(8, 13))
        while True:
            x = r.integers(0, 2, n)
            y = r.integers(0, 2, n)
            a, b, c, d = abcd(x, y)
            if a > 0 and b + c > 0 and d > 0:
                break
        soru = f"İki genotipin bant dizileri: **i = {''.join(map(str, x))}**, **j = {''.join(map(str, y))}**. Jaccard, basit eşleşme ve Dice katsayılarını hesaplayın."
        adim = [f"Karakterleri sayın: a (1-1) = {a}, b (1-0) = {b}, c (0-1) = {c}, d (0-0) = {d}; toplam {n}",
                f"Jaccard = a / (a + b + c) = {a} / {a + b + c} = **{_tr(a / (a + b + c), 3)}**",
                f"Basit eşleşme = (a + d) / n = {a + d} / {n} = **{_tr((a + d) / n, 3)}**",
                f"Dice = 2a / (2a + b + c) = {2 * a} / {2 * a + b + c} = **{_tr(2 * a / (2 * a + b + c), 3)}**",
                f"Yorum: d = {d} ortak sıfır SM'yi yükseltiyor. Baskın markörlerde Jaccard ya da Dice tercih edilir."]
        return soru, adim, f"J = {_tr(a / (a + b + c), 3)}, SM = {_tr((a + d) / n, 3)}, D = {_tr(2 * a / (2 * a + b + c), 3)}"
    if tur == ORNEK_TURLERI[3]:
        s1, s2 = int(r.integers(1, 5)), int(r.integers(1, 5))
        rho = float(r.choice([-0.6, -0.5, 0.5, 0.6, 0.8]))
        s12 = round(rho * math.sqrt(s1 * s2), 2)
        f = r.integers(-3, 4, 2)
        while not np.any(f):
            f = r.integers(-3, 4, 2)
        det = s1 * s2 - s12 ** 2
        inv = np.array([[s2, -s12], [-s12, s1]]) / det
        ara = inv @ f
        d2 = float(f @ ara)
        soru = (f"Kovaryans matrisi S = [[{s1}, {_tr(s12)}], [{_tr(s12)}, {s2}]], iki OTU arasındaki fark vektörü "
                f"x − y = ({f[0]}, {f[1]}). Mahalanobis uzaklığını bulun.")
        adim = [f"det(S) = {s1}×{s2} − {_tr(s12)}² = **{_tr(det, 4)}**",
                f"S⁻¹ = (1/{_tr(det, 4)}) × [[{s2}, {_tr(-s12)}], [{_tr(-s12)}, {s1}]] = [[{_tr(inv[0, 0], 3)}, {_tr(inv[0, 1], 3)}], [{_tr(inv[1, 0], 3)}, {_tr(inv[1, 1], 3)}]]",
                f"S⁻¹(x − y) = ({_tr(ara[0], 3)}, {_tr(ara[1], 3)})",
                f"(x − y)ᵀ S⁻¹ (x − y) = {f[0]}×{_tr(ara[0], 3)} + {f[1]}×{_tr(ara[1], 3)} = **{_tr(d2, 3)}**",
                f"d_M = √{_tr(d2, 3)} = **{_tr(math.sqrt(d2), 3)}** (karşılaştırma: Öklid = {_tr(float(np.hypot(*f)), 3)})"]
        return soru, adim, f"d_M = {_tr(math.sqrt(d2), 3)}"
    if tur in ORNEK_TURLERI[4:6]:
        ad = list("ABCD")
        D = _rastgele_D(r, 4)
        yontemler = ["upgma"] if tur == ORNEK_TURLERI[4] else ["tek", "tam"]
        soru = "Aşağıdaki uzaklık matrisinden " + (" ve ".join(YONTEM_AD[y] for y in yontemler)) + " ile dendrogram kurun.\n\n" + \
               _md_matris(D, ad)
        adim = []
        for y in yontemler:
            ad_, b = hiyerarsik(D, ad, y)
            adim.append(f"**{YONTEM_AD[y]}**")
            for k, a in enumerate(ad_, 1):
                adim.append(f"Adım {k}: en küçük d({a['adlar'][a['i']]}, {a['adlar'][a['j']]}) = {_tr(a['deger'], 3)} → **{a['yeni']}**")
                for hedef, f in a["formul"]:
                    adim.append(f"&nbsp;&nbsp;d({a['yeni']}, {hedef}) = {f}")
            adim.append(f"Birleşme düzeyleri: {', '.join(_tr(x[2], 3) for x in b)} · kofenetik r = {_tr(kofenetik_r(D, b), 3)}")
        return soru, adim, "Adımlar ve dendrogram aşağıda"
    if tur == ORNEK_TURLERI[6]:
        ad = list("ABCD")
        D = _rastgele_D(r, 4)
        soru = "Aşağıdaki uzaklık matrisinden Neighbor-Joining ile ağaç kurun.\n\n" + _md_matris(D, ad)
        ad_, ke = komsu_birlestirme(D, ad)
        adim = []
        for k, a in enumerate(ad_, 1):
            if a.get("son"):
                adim.append(f"Son: {a['adlar'][0]} – {a['adlar'][1]} dalı = {_tr(float(a['D'][0, 1]), 3)}")
                continue
            n = len(a["adlar"])
            adim.append(f"Adım {k} (n = {n}): r = " + ", ".join(f"{x}: {_tr(v, 3)}" for x, v in zip(a["adlar"], a["r"])))
            qs = [f"Q({a['adlar'][i]},{a['adlar'][j]}) = {_tr(a['Q'][i, j], 3)}" for i in range(n) for j in range(i + 1, n)]
            adim.append("&nbsp;&nbsp;" + " · ".join(qs))
            adim.append(f"&nbsp;&nbsp;En küçük Q: **{a['adlar'][a['i']]} + {a['adlar'][a['j']]} → {a['u']}**; dallar "
                        f"{_tr(a['di'], 3)} ve {_tr(a['dj'], 3)}")
            adim.append("&nbsp;&nbsp;Yeni uzaklıklar: " + ", ".join(f"d({a['u']},{h}) = {_tr(v, 3)}" for h, v in a["yeni"]))
        return soru, adim, "Dal uzunlukları: " + ", ".join(f"{x}–{y}: {_tr(l, 3)}" for x, y, l in ke)
    # kofenetik
    ad = list("ABCDE")
    D = _rastgele_D(r, 5)
    _, b = hiyerarsik(D, ad, "upgma")
    C = kofenetik(b, 5)
    i, j = sorted(r.choice(5, 2, replace=False))
    soru = "Aşağıdaki matristen UPGMA ağacı kurulduğunda " + f"**{ad[i]}** ile **{ad[j]}** arasındaki kofenetik uzaklık nedir?\n\n" + _md_matris(D, ad)
    ad_, _ = hiyerarsik(D, ad, "upgma")
    adim = [f"Adım {k}: {a['yeni']} düzey {_tr(a['deger'], 3)}" for k, a in enumerate(ad_, 1)]
    adim.append(f"{ad[i]} ile {ad[j]}'yi ilk kez aynı kümeye koyan birleşmenin düzeyi: **{_tr(C[i, j], 3)}** (özgün uzaklık {_tr(D[i, j], 3)})")
    return soru, adim, f"Kofenetik uzaklık = {_tr(C[i, j], 3)}"


def _md_matris(D, ad):
    s = "| | " + " | ".join(ad) + " |\n|" + "---|" * (len(ad) + 1) + "\n"
    for i, a in enumerate(ad):
        s += f"| **{a}** | " + " | ".join("–" if i == j else _tr(float(D[i, j]), 3) for j in range(len(ad))) + " |\n"
    return s


def sayfa_ornekler():
    st.title("9 · Çözümlü örnekler")
    st.markdown("Bir konu seçin; her **🎲 Yeni örnek** tıklamasında yeni sayılarla bir problem gelir. Önce kâğıt-kalemle çözün, "
                "sonra çözümü açın.")
    tur = st.selectbox("Konu", ORNEK_TURLERI, key="orn_tur")
    tk = f"orn_tohum_{ORNEK_TURLERI.index(tur)}"
    if tk not in st.session_state:
        st.session_state[tk] = int(rng().integers(0, 10 ** 6))
    if st.button("🎲 Yeni örnek", key="orn_yeni"):
        st.session_state[tk] = int(rng().integers(0, 10 ** 6))
    soru, adimlar, cevap = ornek_uret(tur, np.random.default_rng(st.session_state[tk]))
    with st.container(border=True):
        st.markdown("**Problem.** " + soru)
    with st.expander("📖 Adım adım çözüm", expanded=False):
        for k, a in enumerate(adimlar, 1):
            st.markdown(a)
        st.success(f"Sonuç: {cevap}")
    if tur in ORNEK_TURLERI[4:7]:
        with st.expander("🌳 Ağacı göster", expanded=False):
            r = np.random.default_rng(st.session_state[tk])
            D = _rastgele_D(r, 4)
            if tur == ORNEK_TURLERI[6]:
                _, ke = komsu_birlestirme(D, list("ABCD"))
                svg_goster(svg_nj_agac(ke, list("ABCD"), "NJ ağacı"))
            else:
                for y in (["upgma"] if tur == ORNEK_TURLERI[4] else ["tek", "tam"]):
                    _, b = hiyerarsik(D, list("ABCD"), y)
                    svg_goster(svg_dendrogram(b, list("ABCD"), baslik=YONTEM_AD[y]))
    alistirma_blogu("ornek", "Hesap soruları", gen_ornek)


# ---------------------------------------------------------------------------
def _kart_yeni():
    st.session_state.kart_i = int(rng().integers(0, len(TERIMLER)))
    st.session_state.kart_acik = False


def sayfa_tekrar():
    st.title("10 · Genel tekrar")
    st.header("🃏 Hafıza kartları")
    if "kart_i" not in st.session_state:
        _kart_yeni()
    terim, tanim = TERIMLER[st.session_state.kart_i]
    st.markdown(f'<div class="flash">{terim}' + (f'<div class="flash-cevap">{tanim}</div>' if st.session_state.kart_acik else "")
                + '</div>', unsafe_allow_html=True)
    c1, c2, _ = st.columns([1, 1, 2])
    if c1.button("👀 Tanımı göster"):
        st.session_state.kart_acik = True
        st.rerun()
    c2.button("➡️ Sonraki kart", on_click=_kart_yeni)
    with st.expander("📖 Sözlük"):
        ara = st.text_input("Terim ara", "", key="sozluk_ara")
        for t, d in TERIMLER:
            if ara.lower() in (t + d).lower():
                st.markdown(f"**{t}:** {d}")
    st.header("Yöntemin tarihinden")
    for k, a in ONEMLI_KISILER:
        st.markdown(f'<div class="zaman"><b class="yil">{k}</b><br>{a}</div>', unsafe_allow_html=True)
    st.header("Bir bakışta")
    tablo(pd.DataFrame([
        ("Veri ikili, ortak yokluk anlamsız", "Jaccard / Dice", "UPGMA"),
        ("Veri ikili, iki durum eşit anlamlı", "Basit eşleşme", "UPGMA"),
        ("Nicel, birimler farklı", "Standartlaştırılmış Öklid", "UPGMA"),
        ("Nicel, karakterler ilişkili", "Mahalanobis", "UPGMA"),
        ("Ekspresyon profili", "1 − r", "UPGMA"),
        ("DNA dizisi, hızlar farklı olabilir", "Düzeltilmiş dizi uzaklığı", "Neighbor-Joining"),
    ], columns=["Durum", "Ölçü", "Ağaç yöntemi"]))
    alistirma_blogu("tekrar", "Karışık sorular", gen_tekrar)


# ===========================================================================
# SORU ÜRETİCİLERİ
# ===========================================================================
def _secim(r, metin, dogru, yanlis, cozum):
    temiz = [y for y in dict.fromkeys(yanlis) if y != dogru][:3]
    return soru_secim(r, metin, dogru, temiz, [_md(c) if isinstance(c, str) else c for c in cozum])


def _tamamla(r, sabit, havuz):
    gerek = SORU_SAYISI - len(sabit)
    secilen = _kavram_sec(r, havuz, min(max(gerek, 0), len(havuz)))
    hepsi = sabit + secilen
    sira = r.permutation(len(hepsi))
    return [hepsi[i] for i in sira][:max(SORU_SAYISI, len(sabit))]


def _ek(r, liste):
    return [_secim(r, m, d, y, [c]) for m, d, y, c in liste]


def _sec_bir(r, liste):
    return liste[int(r.integers(0, len(liste)))]


def _diger(r, tum, dogru, n=3):
    aday = [x for x in dict.fromkeys(tum) if x != dogru]
    return [aday[int(i)] for i in r.permutation(len(aday))[:n]]


def _vek(v):
    return "(" + "; ".join(_tr(float(x), 2) for x in v) + ")"


# ---------------------------------------------------------------------------
S_NT = [
    ("Numerik taksonominin kurucu kitabı hangisidir?", "Sokal ve Sneath (1963), Principles of Numerical Taxonomy",
     ["Darwin (1859), Türlerin Kökeni", "Mendel (1866), Bitki Melezleri Üzerine Deneyler", "Saitou ve Nei (1987)"], "Numerik taksonomi bu kitapla kuruldu."),
    ("OTU ne demektir?", "Operasyonel Taksonomik Birim", ["Ortak Tür Uzaklığı", "Ortalama Taksonomik Uzaklık", "Organizma Tanımlama Ünitesi"], "Sınıflandırılan birim."),
    ("Veri matrisinin satırlarında genellikle ne bulunur?", "OTU'lar", ["Karakterler", "Uzaklıklar", "Kümeler"], "Satır = OTU, sütun = karakter."),
    ("Veri matrisinin sütunları neyi gösterir?", "Karakterleri (attribute)", ["OTU'ları", "Kümeleri", "Dendrogramı"], "Her sütun bir karakter."),
    ("Veri matrisinin bir hücresindeki değere ne denir?", "Karakter durumu", ["Karakter", "OTU", "Kofenetik uzaklık"], "Karakterin o OTU'daki değeri."),
    ("İngilizce kaynaklardaki 'attribute' teriminin karşılığı hangisidir?", "Karakter (özellik, değişken)", ["OTU", "Küme", "Dal uzunluğu"], "Character, attribute, variable, feature."),
    ("Adanson ilkesi neyi savunur?", "Karakterlere önceden ağırlık vermeden eşit ağırlık verilmesini", ["Tek bir önemli karakterle sınıflandırmayı", "Yalnız moleküler veriyi", "Yalnız ata karakterleri"], "Michel Adanson, 1763."),
    ("Numerik taksonomide neden çok sayıda karakter önerilir?", "Sınıflandırmayı sağlamlaştırmak ve rastlantı etkisini azaltmak", ["Hesabı kısaltmak", "Ağacı köklemek", "Mutasyonu artırmak"], "Tek tek karakterlerin rastlantısal etkisi azalır."),
    ("Fenetik sınıflandırma neye dayanır?", "Genel benzerliğe", ["Yalnız ortak türemiş karakterlere", "Fosil kayıtlarına", "Coğrafyaya"], "Kladistikten farkı."),
    ("Kladistik sınıflandırma neye dayanır?", "Ortak türemiş karakterlere (sinapomorfi)", ["Genel benzerliğe", "Eşit ağırlıklı bütün karakterlere", "Uzaklık matrisine"], "Ata ve türemiş durumlar ayrılır."),
    ("Fenetik yöntemle kurulan dendrograma ne denir?", "Fenogram", ["Kladogram", "Kromatogram", "Histogram"], "Genel benzerlik ağacı."),
    ("'Tane rengi' bir veri matrisinde neyi temsil eder?", "Karakter", ["OTU", "Karakter durumu", "Küme"], "Sütun."),
    ("'Siyah' tane rengi neyi temsil eder?", "Karakter durumu", ["Karakter", "OTU", "Uzaklık"], "Hücredeki değer."),
    ("Çalışmaya alınan 30 zeytin genotipinin her biri nedir?", "OTU", ["Karakter", "Karakter durumu", "Dış grup"], "Sınıflandırılan birim."),
    ("Bant var / yok hangi karakter türüdür?", "İkili (iki durumlu)", ["Sıralı", "Nicel sürekli", "Çok durumlu nitel"], "0/1."),
    ("'Tüylülük: yok < seyrek < yoğun' hangi karakter türüdür?", "Sıralı (ordinal)", ["İkili", "Sırasız nitel", "Nicel sürekli"], "Durumlar arasında sıra var."),
    ("'Çiçek tipi: erkek / dişi / erselik' hangi karakter türüdür?", "Çok durumlu nitel (sırasız)", ["Sıralı", "İkili", "Nicel"], "Sıra yok."),
    ("Tane ağırlığı (g) hangi karakter türüdür?", "Nicel sürekli", ["Nicel kesikli", "İkili", "Sıralı"], "Ölçülür."),
    ("Yaprak lob sayısı hangi karakter türüdür?", "Nicel kesikli", ["Nicel sürekli", "İkili", "Sırasız nitel"], "Sayılır."),
    ("Sırasız nitel bir karakteri 1, 2, 3 diye tek sütunda kodlamanın sakıncası nedir?", "Durumlar arasında olmayan bir sıra ve uzaklık varsayılır", ["Hesap yapılamaz", "Jaccard tanımsız olur", "Hiçbir sakıncası yoktur"], "Ayrı ikili sütunlar ya da eşleşme katsayısı kullanılır."),
    ("Numerik taksonomide çalışma sırası hangisidir?", "OTU ve karakter seçimi → veri matrisi → benzerlik → kümeleme → fenogram", ["Kümeleme → veri matrisi → OTU seçimi", "Fenogram → benzerlik → veri matrisi", "Benzerlik → OTU seçimi → kümeleme"], "Akış şeması."),
    ("Nicel karakterlerin birimleri farklıysa uzaklıktan önce ne yapılır?", "Standartlaştırma", ["Kofenetik korelasyon", "Köklendirme", "Bootstrap"], "z-skoru ya da aralık."),
    ("Numerik taksonominin bir üstünlüğü hangisidir?", "Aynı veri ve yöntemle tekrarlanabilir sonuç", ["Evrimsel soyu kesin gösterir", "Karakter gerektirmez", "Tek karakter yeterlidir"], "Nesnellik."),
    ("Karışık (nicel + nitel + ikili) veri için hangi katsayı kullanılır?", "Gower", ["Jaccard", "Pearson r", "Mahalanobis"], "Her karakter 0–1'e getirilir."),
    ("Bir veri matrisi 25 OTU ve 60 karakterden oluşuyorsa kaç hücresi vardır?", "1500", ["85", "60", "625"], "25 × 60."),
    ("Bir OTU neden 'operasyonel' diye adlandırılır?", "Çalışmanın amacına göre araştırıcı tarafından belirlendiği için", ["Ameliyatla elde edildiği için", "Her zaman tür olduğu için", "Yazılım belirlediği için"], "Birey, çeşit, popülasyon olabilir."),
    ("UPGMA ve basit eşleşme katsayısı kime aittir?", "Sokal ve Michener (1958)", ["Saitou ve Nei (1987)", "Mahalanobis (1936)", "Jaccard (1901)"], "Numerik taksonominin temel taşları."),
    ("Hangisi bir OTU olamaz?", "Tane ağırlığı", ["Bir asma çeşidi", "Bir popülasyon", "Bir herbaryum örneği"], "Tane ağırlığı bir karakterdir."),
    ("Hangisi bir karakter değildir?", "Bir zeytin popülasyonu", ["Yaprak boyu", "ISSR bandı", "Çiçek rengi"], "Popülasyon OTU olur."),
    ("NTSYSpc gibi bazı programlar matrisi nasıl ister?", "Karakterler satırda, OTU'lar sütunda (devrik)", ["Yalnız uzaklık olarak", "Yalnız ikili", "Yalnız tek satır"], "Hesap aynıdır, yön değişir."),
]

S_YAKIN = [
    ("Jaccard benzerliği 0,95 ise iki OTU için ne söylenir?", "Çok benzerler", ["Çok farklılar", "Ters örüntüdeler", "Hesaplanamaz"], "1'e yakın benzerlik."),
    ("Jaccard uzaklığı 0,95 ise iki OTU için ne söylenir?", "Çok farklılar", ["Çok benzerler", "Özdeşler", "Ters örüntüdeler"], "Uzaklık 1'e yakın."),
    ("Öklid uzaklığı 0 ise ne söylenir?", "İki OTU bütün karakterlerde özdeştir", ["Hiç ortak karakter yoktur", "Ters örüntüdedirler", "Hesap hatası vardır"], "Özdeşlik."),
    ("Öklid uzaklığının üst sınırı nedir?", "Yoktur; birime ve karakter sayısına bağlıdır", ["1", "2", "100"], "0 – ∞."),
    ("Pearson korelasyonu −1 ise iki ekspresyon profili için ne söylenir?", "Tam ters örüntü", ["Özdeş", "İlişkisiz", "Çok benzer"], "Biri artarken öteki azalır."),
    ("Korelasyon uzaklığı (1 − r) hangi aralıktadır?", "0 – 2", ["0 – 1", "−1 – 1", "0 – ∞"], "r = −1 → 2."),
    ("Bir ölçünün 'metrik' olması için hangisi gerekmez?", "Değerlerin 0 ile 1 arasında olması", ["Simetri", "Üçgen eşitsizliği", "Negatif olmama"], "Öklid 1'i aşabilir ve metriktir."),
    ("Üçgen eşitsizliği hangisidir?", "d(A,C) ≤ d(A,B) + d(B,C)", ["d(A,B) = d(B,A)", "d(A,A) = 0", "d(A,C) ≥ d(A,B) × d(B,C)"], "Dolambaçlı yol daha kısa olamaz."),
    ("Simetri koşulu hangisidir?", "d(A,B) = d(B,A)", ["d(A,C) ≤ d(A,B) + d(B,C)", "d ≥ 0", "d(A,A) = 1"], "Sıra fark etmez."),
    ("Ultrametrik koşul hangisidir?", "d(A,C) ≤ maks[d(A,B), d(B,C)]", ["d(A,C) ≤ d(A,B) + d(B,C)", "d(A,B) = d(B,A)", "d(A,C) = d(A,B) − d(B,C)"], "UPGMA ağaçları ultrametriktir."),
    ("Üçgen eşitsizliğini sağlamayan ama öteki koşulları sağlayan ölçüye ne denir?", "Yarı metrik (semimetrik)", ["Ultrametrik", "Öklid", "Toplamsal"], "Bray–Curtis örneği."),
    ("Hangisi metrik değildir?", "Kare Öklid uzaklığı", ["Öklid", "Manhattan", "1 − Jaccard"], "Üçgen eşitsizliği bozulabilir."),
    ("Bray–Curtis uzaklığı hangisidir?", "Yarı metrik", ["Metrik", "Ultrametrik", "Benzerlik katsayısı"], "Üçgen eşitsizliğini her zaman sağlamaz."),
    ("Benzerlik ile uzaklık arasındaki ilişki nasıldır?", "Biri artarken öteki azalır", ["İkisi birlikte artar", "İlişkisizdir", "Hep eşittir"], "d = 1 − s."),
    ("Dice uzaklığı (1 − D) nasıl bir ölçüdür?", "Yarı metrik", ["Metrik", "Ultrametrik", "Benzerlik"], "Sørensen uzaklığı da yarı metriktir."),
    ("Bir değerin '0'a yakın' olmasını yorumlamadan önce ne bilinmelidir?", "Ölçünün benzerlik mi uzaklık mı olduğu", ["Karakter sayısı", "Yazılımın adı", "OTU'ların adları"], "Anlam ters döner."),
    ("Bray–Curtis = 1 ne anlama gelir?", "İki alanın hiç ortak türü yoktur", ["İki alan özdeştir", "Bolluklar eşittir", "Hesaplanamaz"], "En büyük farklılık."),
    ("Basit eşleşme katsayısı 1 ise ne söylenir?", "Bütün karakterlerde aynı durumdalar", ["Hiç ortak bant yok", "Ters örüntü", "Yarı benzer"], "Özdeş."),
    ("Hangi dönüşüm bazı katsayılara Öklid uzayında çizilebilme özelliği kazandırır?", "d = √(1 − s)", ["d = 1 + s", "d = s²", "d = log s"], "PCoA'da tercih edilir."),
    ("d(A,A) = 0 koşulunun adı nedir?", "Özdeşlik (öz-uzaklık sıfır)", ["Simetri", "Üçgen eşitsizliği", "Ultrametrik"], "Kendisine uzaklık sıfır."),
]

S_OKLID = [
    ("Öklid uzaklığı nasıl hesaplanır?", "Farkların karelerinin toplamının karekökü", ["Mutlak farkların toplamı", "Farkların en büyüğü", "Ortak bantların oranı"], "√Σ(xᵢ − yᵢ)²."),
    ("Manhattan uzaklığı nasıl hesaplanır?", "Mutlak farkların toplamı", ["Farkların kareleri toplamının karekökü", "Kovaryansla düzeltilmiş fark", "1 − Jaccard"], "Σ|xᵢ − yᵢ|."),
    ("Aynı iki OTU için Manhattan ile Öklid arasındaki ilişki nedir?", "Manhattan ≥ Öklid", ["Manhattan < Öklid", "Her zaman eşittir", "İlişki yoktur"], "Bloklar boyunca yol daha uzun ya da eşittir."),
    ("Birimleri farklı karakterlerde ham Öklid uzaklığının sorunu nedir?", "Sayısal değeri büyük karakter sonucu tek başına belirler", ["Hesaplanamaz", "Hep 0 çıkar", "Negatif olur"], "Salkım ağırlığı (g) örneği."),
    ("z-skoru standartlaştırması nasıl yapılır?", "(x − ortalama) / standart sapma", ["x / en büyük değer", "x − en küçük değer", "x²"], "Her karakter ortalama 0, sd 1."),
    ("Aralık standartlaştırmasında değerler hangi aralığa getirilir?", "0 – 1", ["−1 – 1", "0 – 100", "−3 – 3"], "(x − min) / (maks − min)."),
    ("Salkım ağırlığını g yerine kg ile yazmak standartlaştırılmamış Öklid sonucunu değiştirir mi?", "Evet", ["Hayır", "Yalnız Jaccard değişir", "Yalnız ağaç kökü değişir"], "Birim değişince karakterin payı değişir."),
    ("Birim değiştirmek z-skoru ile standartlaştırılmış Öklid sonucunu değiştirir mi?", "Hayır", ["Evet", "Yalnız ikili veride", "Yalnız 2 karakterde"], "z-skoru birimden bağımsızdır."),
    ("Kare Öklid uzaklığı hangisidir?", "Σ(xᵢ − yᵢ)²", ["√Σ(xᵢ − yᵢ)²", "Σ|xᵢ − yᵢ|", "maks|xᵢ − yᵢ|"], "Karekök alınmadan."),
    ("Öklid uzaklığı hangi veri türü için uygundur?", "Nicel (ölçülen) veri", ["Var/yok bant verisi", "Sırasız nitel veri", "DNA dizileri"], "Ölçümler."),
    ("Öklid uzaklığı neden 'kuş uçuşu' uzaklık olarak anılır?", "İki nokta arasındaki düz çizginin uzunluğu olduğu için", ["Kuş verisinde kullanıldığı için", "Bloklar boyunca ölçüldüğü için", "Kovaryans kullandığı için"], "Pisagor."),
    ("Standartlaştırma numerik taksonominin hangi ilkesiyle uyumludur?", "Karakterlere eşit ağırlık", ["Tek karakterle sınıflandırma", "Ata karakterlerin seçimi", "Moleküler saat"], "Adanson ilkesi."),
]

S_MAHAL = [
    ("Mahalanobis uzaklığının Öklid'den farkı nedir?", "Değişkenlerin varyans ve korelasyonunu hesaba katar", ["Yalnız ikili veride kullanılır", "Mutlak farkları toplar", "Ortak sıfırları yok sayar"], "S⁻¹ ile ağırlıklandırır."),
    ("Mahalanobis uzaklığını kim tanımladı?", "P. C. Mahalanobis (1936)", ["Paul Jaccard (1901)", "Sokal ve Michener (1958)", "Saitou ve Nei (1987)"], "Hint istatistikçi."),
    ("Kovaryans matrisi birim matris ise Mahalanobis neye eşittir?", "Öklid uzaklığına", ["Manhattan uzaklığına", "Jaccard uzaklığına", "0'a"], "S = I."),
    ("Kovaryans matrisi köşegen ise (korelasyon yok) Mahalanobis neye eşittir?", "Standartlaştırılmış Öklid uzaklığına", ["Manhattan uzaklığına", "Bray–Curtis'e", "1 − r'ye"], "Her fark kendi sd'sine bölünür."),
    ("Kovaryans matrisinin köşegeninde ne bulunur?", "Varyanslar", ["Kovaryanslar", "Korelasyonlar", "Ortalamalar"], "Köşegen dışı: kovaryanslar."),
    ("Mahalanobis hesabında hangi matrisin tersi kullanılır?", "Kovaryans matrisi", ["Uzaklık matrisi", "Veri matrisi", "Q matrisi"], "S⁻¹."),
    ("Pozitif korelasyonlu iki karakterde, korelasyon yönündeki bir nokta ile ters yöndeki bir nokta merkeze aynı Öklid uzaklığındaysa hangisi Mahalanobis'e göre daha uzaktır?", "Ters yöndeki nokta", ["Korelasyon yönündeki nokta", "İkisi eşittir", "Karar verilemez"], "Ters yönde sapma 'olağan dışı'dır."),
    ("Mahalanobis uzaklığının kullanıldığı bir alan hangisidir?", "Aykırı gözlem (outlier) bulma", ["Bant sayma", "Dizi hizalama", "PCR"], "Ayrıca diskriminant analizi."),
    ("Mahalanobis uzaklığının bir sınırlaması hangisidir?", "Kovaryans matrisinin tersinin alınabilmesi gerekir", ["İkili veride çok iyi çalışır", "Korelasyonu dikkate almaz", "Birime çok duyarlıdır"], "n > p ve tekrarlı değişken olmamalı."),
    ("Mahalanobis uzaklığı birim değişikliğinden (cm → mm) etkilenir mi?", "Hayır", ["Evet", "Yalnız 2 karakterde", "Yalnız negatif değerlerde"], "Kovaryans aynı oranda değişir."),
    ("Mahalanobis eş-uzaklık eğrileri iki karakterde hangi biçimdedir?", "Elips", ["Kare", "Düz çizgi", "Üçgen"], "Korelasyon yönünde uzanır."),
    ("2 × 2 matris [[a, b], [b, c]]'nin determinantı nedir?", "ac − b²", ["a + c", "ab − c", "a² + c²"], "Ters için gerekir."),
    ("[[a, b], [b, c]] matrisinin tersi hangisidir?", "(1 / (ac − b²)) × [[c, −b], [−b, a]]", ["[[1/a, 1/b], [1/b, 1/c]]", "[[c, b], [b, a]]", "[[a, −b], [−b, c]]"], "Köşegen yer değiştirir, işaret değişir."),
]

S_IKILI = [
    ("Jaccard katsayısının formülü hangisidir?", "a / (a + b + c)", ["(a + d) / (a + b + c + d)", "2a / (2a + b + c)", "d / (a + d)"], "Ortak sıfırlar yok."),
    ("Basit eşleşme katsayısının formülü hangisidir?", "(a + d) / (a + b + c + d)", ["a / (a + b + c)", "2a / (2a + b + c)", "a / n"], "Ortak var + ortak yok."),
    ("Dice katsayısının formülü hangisidir?", "2a / (2a + b + c)", ["a / (a + b + c)", "(a + d) / n", "a / (b + c)"], "Ortak varlığa iki kat ağırlık."),
    ("a, b, c, d tablosunda 'd' neyi gösterir?", "İki OTU'da da 0 olan karakter sayısı", ["İki OTU'da da 1 olan", "Yalnız i'de 1 olan", "Yalnız j'de 1 olan"], "Ortak yokluk."),
    ("a, b, c, d tablosunda 'a' neyi gösterir?", "İki OTU'da da 1 olan karakter sayısı", ["İki OTU'da da 0 olan", "Yalnız i'de 1 olan", "Toplam karakter"], "Ortak varlık."),
    ("Baskın markörlerde (ISSR, RAPD) neden Jaccard tercih edilir?", "Ortak bant yokluğu benzerlik kanıtı değildir", ["Daha hızlı hesaplanır", "Negatif değer verir", "Kovaryansı dikkate alır"], "0-0 eşleşmeleri yok sayılır."),
    ("Ortak 0-0 karakter sayısı artınca Jaccard ne olur?", "Değişmez", ["Artar", "Azalır", "1 olur"], "d formülde yok."),
    ("Ortak 0-0 karakter sayısı artınca basit eşleşme ne olur?", "Artar", ["Değişmez", "Azalır", "0 olur"], "d paydada ve payda."),
    ("Jaccard ile Dice arasındaki ilişki nedir?", "D = 2J / (1 + J)", ["D = J²", "D = 1 − J", "D = J / 2"], "Sıralama aynıdır."),
    ("Dice ile Jaccard karşılaştırılınca hangisi doğrudur?", "Dice ≥ Jaccard", ["Dice < Jaccard", "Her zaman eşit", "İlişkisiz"], "a'ya iki kat ağırlık."),
    ("Basit eşleşme ile Jaccard karşılaştırılınca hangisi doğrudur?", "SM ≥ Jaccard", ["SM < Jaccard", "Hep eşit", "İlişkisiz"], "d = 0 ise eşit."),
    ("Jaccard katsayısı ilk olarak ne için geliştirildi?", "Alpler'deki bitki floralarını karşılaştırmak için", ["DNA dizilerini hizalamak için", "Kan gruplarını sınıflamak için", "Aykırı gözlem bulmak için"], "Paul Jaccard, 1901."),
    ("Sørensen katsayısı hangi katsayıyla aynıdır?", "Dice", ["Jaccard", "Basit eşleşme", "Gower"], "Ekolojide Sørensen adıyla."),
    ("Basit eşleşme hangi durumda uygundur?", "0 ve 1 eşit anlamlıysa (simetrik ikili karakter)", ["Baskın markörlerde", "Tür listelerinde", "Hiçbir zaman"], "Örn. tüylü/tüysüz."),
    ("Jaccard uzaklığı nasıl elde edilir?", "1 − J", ["1 + J", "J²", "√J"], "Metrik bir uzaklık."),
]

S_UPGMA = [
    ("UPGMA'nın açılımı nedir?", "Unweighted Pair Group Method with Arithmetic mean", ["Unrooted Phylogenetic Group Mean Analysis", "Universal Pair Grouping Matrix Algorithm", "Unweighted Parsimony Group Method"], "Aritmetik ortalamalı ağırlıksız çift grup."),
    ("Aglomeratif kümeleme her adımda neyi birleştirir?", "Uzaklık matrisindeki en küçük değere sahip iki kümeyi", ["En büyük uzaklıktaki iki kümeyi", "Rastgele iki kümeyi", "Q değeri en büyük çifti"], "Birleştirici yöntem."),
    ("n OTU için hiyerarşik kümeleme kaç birleşme adımı gerektirir?", "n − 1", ["n", "n + 1", "n²"], "Sonunda tek küme kalır."),
    ("Tek bağlantıda yeni uzaklık nasıl hesaplanır?", "min[d(i,K), d(j,K)]", ["maks[d(i,K), d(j,K)]", "Ağırlıklı ortalama", "Q ölçütüyle"], "En yakın komşu."),
    ("Tam bağlantıda yeni uzaklık nasıl hesaplanır?", "maks[d(i,K), d(j,K)]", ["min[d(i,K), d(j,K)]", "Ortalama", "d(i,j) / 2"], "En uzak komşu."),
    ("Zincirleme (chaining) eğilimi en çok hangi yöntemde görülür?", "Tek bağlantı", ["Tam bağlantı", "UPGMA", "NJ"], "Kümeler zincir gibi büyür."),
    ("UPGMA'da kümeler arası uzaklık nedir?", "Üyeler arasındaki bütün uzaklıkların ortalaması", ["En küçük uzaklık", "En büyük uzaklık", "Merkezler arası Mahalanobis"], "Ortalama bağlantı."),
    ("UPGMA hangi tür ağaç üretir?", "Ultrametrik (yapraklar kökten eşit uzaklıkta)", ["Köksüz toplamsal", "Kladogram", "Rastgele"], "Moleküler saat varsayımı."),
    ("Dendrogramda yan yana çizilmiş iki OTU için ne söylenebilir?", "Birleşme düzeyine bakılmadan benzerlik çıkarılamaz", ["Mutlaka en benzer çifttir", "Mutlaka farklıdır", "Aynı türdür"], "Dallar döndürülebilir."),
    ("Kofenetik uzaklık nedir?", "İki OTU'nun dendrogramda ilk kez birleştiği düzey", ["Özgün uzaklık", "Dal sayısı", "Q değeri"], "Ağaçtan okunur."),
    ("Kofenetik korelasyon neyi ölçer?", "Ağacın özgün uzaklıkları ne kadar iyi yansıttığını", ["Karakter sayısını", "Bant sayısını", "Kök yerini"], "Özgün d ile kofenetik d arasındaki r."),
    ("Kofenetik korelasyon r = 0,93 ise uyum nasıldır?", "Çok iyi", ["Zayıf", "Çok zayıf", "Hesap hatası"], "r ≥ 0,9."),
    ("Kofenetik korelasyon r = 0,65 ise uyum nasıldır?", "Çok zayıf", ["Çok iyi", "İyi", "Mükemmel"], "r < 0,7."),
    ("Dendrogramı belirli bir düzeyden kesmek ne sağlar?", "OTU'ları gruplara (kümelere) ayırır", ["Ağacı köklendirir", "Uzaklıkları değiştirir", "Karakter ekler"], "Kesim."),
    ("Filogenetik UPGMA ağacında bir dalın uzunluğu birleşme düzeyiyle nasıl ilişkilidir?", "Düzeyin yarısıdır (d/2)", ["Düzeyin iki katıdır", "Düzeye eşittir", "İlişkisizdir"], "Yaprak kökten d/2 uzaklıkta."),
    ("Kofenetik korelasyonu öneren araştırıcılar kimlerdir?", "Sokal ve Rohlf (1962)", ["Saitou ve Nei (1987)", "Jaccard (1901)", "Mahalanobis (1936)"], "Taxon dergisi."),
]

S_NJ = [
    ("Neighbor-Joining yöntemini kim geliştirdi?", "Saitou ve Nei (1987)", ["Sokal ve Michener (1958)", "Sokal ve Sneath (1963)", "Jaccard (1901)"], "Mol. Biol. Evol."),
    ("NJ her adımda hangi çifti birleştirir?", "Q değeri en küçük olan çifti", ["Uzaklığı en küçük çifti", "Uzaklığı en büyük çifti", "Rastgele bir çifti"], "Q = (n−2)d − rᵢ − rⱼ."),
    ("NJ'de Q(i,j) formülü hangisidir?", "(n − 2)·d(i,j) − rᵢ − rⱼ", ["d(i,j) − rᵢ − rⱼ", "(rᵢ + rⱼ) / 2", "n·d(i,j) + rᵢ"], "rᵢ: satır toplamı."),
    ("NJ'de rᵢ neyi gösterir?", "i'nin öteki bütün OTU'lara uzaklıklarının toplamı", ["i'nin dal uzunluğu", "i'nin karakter sayısı", "i'nin Q değeri"], "Satır toplamı."),
    ("NJ hangi tür ağaç verir?", "Köksüz, toplamsal ağaç", ["Köklü, ultrametrik ağaç", "Kladogram", "Fenogram"], "Köklemek için dış grup."),
    ("NJ'nin UPGMA'ya göre üstünlüğü nedir?", "Değişim hızlarının eşit olmasını varsaymaz", ["Daha az hesap gerektirir", "Köklü ağaç verir", "Yalnız ikili veride çalışır"], "Saat varsayımı yok."),
    ("NJ'de yeni düğüm u'nun k'ya uzaklığı nasıl hesaplanır?", "[d(i,k) + d(j,k) − d(i,j)] / 2", ["[d(i,k) + d(j,k)] / 2", "min[d(i,k), d(j,k)]", "d(i,k) × d(j,k)"], "Formül."),
    ("NJ'de birleşen i için dal uzunluğu nedir?", "d(i,j)/2 + (rᵢ − rⱼ) / [2(n − 2)]", ["d(i,j)/2", "rᵢ / n", "Q(i,j) / 2"], "δᵢ."),
    ("NJ'de negatif dal uzunluğu çıkarsa ne yapılır?", "0 kabul edilir", ["Ağaç atılır", "Mutlak değeri alınır ve iki katı yazılır", "UPGMA'ya geçilir"], "Veri tam toplamsal değil."),
    ("NJ ağacını köklemenin yaygın yolu nedir?", "Dış grup eklemek", ["En uzun dalı silmek", "Ağacı ters çevirmek", "Q matrisini toplamak"], "Ya da orta noktadan köklenir."),
    ("Toplamsal (additive) ağaç ne demektir?", "Yapraklar arası yol uzunluğu özgün uzaklığa eşittir", ["Bütün dallar eşittir", "Yapraklar kökten eşit uzaklıktadır", "Dallar toplanmaz"], "Dört nokta koşulu."),
    ("NJ en çok hangi verilerle kullanılır?", "DNA / protein dizilerinden elde edilen uzaklıklar", ["Yalnız morfolojik ölçümler", "Yalnız var/yok verisi", "Hiçbiri"], "Filogenetik uzaklık ağaçları."),
    ("NJ bir adımda kaç düğümü birleştirir?", "2", ["3", "1", "n"], "Komşu çift."),
    ("4 OTU'dan oluşan bir NJ ağacında kaç iç düğüm vardır?", "2", ["1", "3", "4"], "Köksüz ikili ağaçta n − 2."),
]

S_BIYO = [
    ("ISSR bant verisi için hangi ölçü uygundur?", "Jaccard ya da Dice", ["Öklid (ham)", "Mahalanobis", "Korelasyon uzaklığı"], "Baskın markör."),
    ("Gen ekspresyon profillerini örüntüye göre gruplamak için hangi ölçü uygundur?", "Korelasyon uzaklığı (1 − r)", ["Jaccard", "Basit eşleşme", "Bray–Curtis"], "Birlikte düzenlenme."),
    ("Farklı birimlerdeki morfolojik ölçümler için hangisi uygundur?", "Standartlaştırılmış Öklid", ["Ham Öklid", "Jaccard", "Dice"], "Önce z-skoru."),
    ("İlişkili morfometrik karakterlerde popülasyonlar arası uzaklık için hangisi uygundur?", "Mahalanobis", ["Jaccard", "Basit eşleşme", "Bray–Curtis"], "Kovaryansı dikkate alır."),
    ("Tür bolluk (birey sayısı) verisi için yaygın ölçü hangisidir?", "Bray–Curtis", ["Öklid", "Mahalanobis", "Basit eşleşme"], "Ortak sıfırları yok sayar."),
    ("Tür var/yok listeleriyle alanları karşılaştırmak için hangisi uygundur?", "Jaccard ya da Sørensen", ["Basit eşleşme", "Öklid", "Mahalanobis"], "Ortak yokluk anlamsız."),
    ("Karışık veri (nicel + nitel + ikili) için hangisi uygundur?", "Gower katsayısı", ["Jaccard", "Pearson r", "Bray–Curtis"], "Her karakter 0–1."),
    ("DNA dizileri arasındaki uzaklık için hangisi kullanılır?", "p-uzaklığı ya da Jukes–Cantor", ["Jaccard", "Bray–Curtis", "Mahalanobis"], "Farklı nükleotit oranı."),
    ("Öklid ile korelasyon uzaklığı ekspresyonda neden farklı ağaç verir?", "Öklid düzey farkına, korelasyon örüntüye bakar", ["Öklid yalnız ikili veride çalışır", "Korelasyon her zaman 0'dır", "İkisi aynıdır"], "Gen1–Gen2 örneği."),
    ("Aynı biçimde artan ama düzeyleri çok farklı iki gen hangi ölçüde yakın çıkar?", "Korelasyon uzaklığı", ["Öklid", "Manhattan", "Bray–Curtis"], "Örüntü aynı."),
    ("Kodominant SSR verisi için hangisi uygundur?", "Allel paylaşımı ya da Nei uzaklığı", ["Ham Öklid", "Bray–Curtis", "Mahalanobis"], "Allel bilgisi kullanılır."),
    ("Bant yokluğunun nedenlerinden biri değildir hangisi?", "İki bireyin aynı alleli taşıması", ["Primer bağlanma bölgesinde mutasyon", "Delesyon", "Amplifikasyon başarısızlığı"], "Bu yüzden 0-0 kanıt değildir."),
    ("Iris veri setinde dört karakter hangileridir?", "Çanak ve taç yaprağı boyu ve eni", ["Tohum sayısı ve rengi", "Kök boyu ve ağırlığı", "Yaprak sayısı ve rengi"], "Anderson 1935; Fisher 1936."),
    ("Ekolojide iki alanda da bulunmayan türler neden benzerlik sayılmaz?", "Bir türün yokluğu birçok nedenden olabilir; alanların ortak özelliğini göstermez",
     ["Hesabı zorlaştırdığı için", "Her zaman sayılır", "Bray–Curtis negatif olduğu için"], "Ortak sıfır sorunu."),
    ("Bir genin profili ötekinin tam iki katıysa korelasyon uzaklığı kaçtır?", "0", ["1", "2", "0,5"], "r = 1."),
    ("Korelasyon uzaklığı neden ekspresyonda yaygındır?", "Örüntü benzerliğini (birlikte düzenlenmeyi) yakalar", ["Düzey farkını büyütür", "İkili veri ister", "Kovaryans gerektirmez diye"], "Ko-ekspresyon."),
    ("Bray–Curtis hangi değerler arasındadır?", "0 – 1", ["−1 – 1", "0 – ∞", "0 – 2"], "Pozitif bolluklarla."),
    ("Iris veri setini istatistik literatürüne kim kazandırdı?", "R. A. Fisher (1936)", ["Mendel (1866)", "Saitou (1987)", "Jaccard (1901)"], "Ölçümleri Edgar Anderson aldı."),
    ("Ham Öklid ile morfolojik veride hangi karakter sonucu belirler?", "Sayısal değeri en büyük (ör. salkım ağırlığı)", ["En küçük değerli karakter", "Hepsi eşit katkı yapar", "Rastgele bir karakter"], "Standartlaştırma gerekir."),
]


# ---------------------------------------------------------------------------
def _s_oklid(r):
    t = int(r.integers(2, 5))
    x = r.integers(0, 10, t)
    y = r.integers(0, 10, t)
    while np.all(x == y):
        y = r.integers(0, 10, t)
    f = y - x
    top = int(np.sum(f ** 2))
    return soru_sayi(f"i = {_vek(x)} ve j = {_vek(y)} için Öklid uzaklığı kaçtır?", math.sqrt(top),
                     [_md(f"Farklar: {', '.join(map(str, f))}; kareler toplamı = {top}; √{top} = **{_tr(math.sqrt(top), 3)}**")],
                     [float(np.sum(np.abs(f))), float(top), math.sqrt(top) / t], ondalik=3)


def _s_manhattan(r):
    t = int(r.integers(2, 6))
    x = r.integers(0, 10, t)
    y = r.integers(0, 10, t)
    while np.all(x == y):
        y = r.integers(0, 10, t)
    f = np.abs(y - x)
    return soru_sayi(f"i = {_vek(x)} ve j = {_vek(y)} için Manhattan uzaklığı kaçtır?", float(np.sum(f)),
                     [_md(f"Mutlak farklar: {', '.join(map(str, f))}; toplam = **{int(np.sum(f))}**")],
                     [math.sqrt(float(np.sum(f ** 2))), float(np.sum(f ** 2)), float(np.max(f))], ondalik=3)


def _s_z(r):
    ort = int(r.integers(10, 60))
    sd = int(r.integers(2, 9))
    x = ort + int(r.integers(-3, 4)) * sd // 2 + int(r.integers(-2, 3))
    z = (x - ort) / sd
    return soru_sayi(f"Bir karakterin ortalaması {ort}, standart sapması {sd}. x = {x} olan OTU'nun z-skoru kaçtır?", z,
                     [_md(f"z = ({x} − {ort}) / {sd} = **{_tr(z, 3)}**")], [x - ort, (x - ort) * sd, x / sd], ondalik=3)


def _s_aralik(r):
    mn = int(r.integers(0, 30))
    mx = mn + int(r.integers(10, 60))
    x = int(r.integers(mn, mx + 1))
    v = (x - mn) / (mx - mn)
    return soru_sayi(f"Bir karakter en küçük {mn}, en büyük {mx}. x = {x} aralık standartlaştırmasında kaç olur?", v,
                     [_md(f"({x} − {mn}) / ({mx} − {mn}) = **{_tr(v, 3)}**")], [x / mx, (x - mn) / mx, (mx - x) / (mx - mn)], ondalik=3)


def _s_ikili(r, hangi):
    n = int(r.integers(8, 15))
    while True:
        x = r.integers(0, 2, n)
        y = r.integers(0, 2, n)
        a, b, c, d = abcd(x, y)
        if a > 0 and b + c > 0 and d > 0:
            break
    J, SM, D = a / (a + b + c), (a + d) / n, 2 * a / (2 * a + b + c)
    ad, deg, form = {"J": ("Jaccard benzerliği", J, f"{a} / ({a} + {b} + {c})"),
                     "SM": ("basit eşleşme katsayısı", SM, f"({a} + {d}) / {n}"),
                     "D": ("Dice katsayısı", D, f"2×{a} / (2×{a} + {b} + {c})"),
                     "JD": ("Jaccard uzaklığı (1 − J)", 1 - J, f"1 − {a}/({a} + {b} + {c})")}[hangi]
    yanlis = [v for v in (J, SM, D, 1 - J, 1 - SM) if abs(v - deg) > 1e-9]
    return soru_sayi(f"İki genotipin bant dizileri: i = {''.join(map(str, x))}, j = {''.join(map(str, y))}. {ad.capitalize()} kaçtır?", deg,
                     [_md(f"a = {a}, b = {b}, c = {c}, d = {d}. {ad.capitalize()} = {form} = **{_tr(deg, 3)}**")], yanlis, ondalik=3)


def _s_abcd(r):
    a, b, c, d = (int(r.integers(1, 9)) for _ in range(4))
    hangi = _sec_bir(r, ["J", "SM", "D"])
    deg = {"J": a / (a + b + c), "SM": (a + d) / (a + b + c + d), "D": 2 * a / (2 * a + b + c)}[hangi]
    ad = {"J": "Jaccard", "SM": "basit eşleşme", "D": "Dice"}[hangi]
    yan = [a / (a + b + c), (a + d) / (a + b + c + d), 2 * a / (2 * a + b + c), a / (a + b + c + d)]
    return soru_sayi(f"a = {a}, b = {b}, c = {c}, d = {d} ise {ad} katsayısı kaçtır?", deg,
                     [_md(f"Formülü uygulayın: **{_tr(deg, 3)}**")], [v for v in yan if abs(v - deg) > 1e-9], ondalik=3)


def _s_mahal(r):
    s1, s2 = int(r.integers(1, 5)), int(r.integers(1, 5))
    f = r.integers(-3, 4, 2)
    while not np.any(f):
        f = r.integers(-3, 4, 2)
    if r.random() < 0.5:
        d2 = f[0] ** 2 / s1 + f[1] ** 2 / s2
        return soru_sayi(f"Korelasyonsuz iki karakterin varyansları {s1} ve {s2}. Fark vektörü ({f[0]}, {f[1]}) ise Mahalanobis uzaklığı kaçtır?",
                         math.sqrt(d2), [_md(f"S köşegen: d² = {f[0]}²/{s1} + {f[1]}²/{s2} = {_tr(d2, 3)}; d = **{_tr(math.sqrt(d2), 3)}**")],
                         [math.hypot(*f), d2, math.sqrt(f[0] ** 2 * s1 + f[1] ** 2 * s2)], ondalik=3)
    rho = float(r.choice([0.5, -0.5]))
    s12 = rho * math.sqrt(s1 * s2)
    S = np.array([[s1, s12], [s12, s2]])
    dm = mahalanobis(f, [0, 0], S)
    return soru_sayi(f"S = [[{s1}, {_tr(s12, 3)}], [{_tr(s12, 3)}, {s2}]] ve fark vektörü ({f[0]}, {f[1]}) ise Mahalanobis uzaklığı kaçtır?",
                     dm, [_md(f"det = {_tr(s1 * s2 - s12 ** 2, 3)}; S⁻¹ ile (x − y)ᵀS⁻¹(x − y) = {_tr(dm ** 2, 3)}; d = **{_tr(dm, 3)}**")],
                     [math.hypot(*f), dm ** 2, math.sqrt(f[0] ** 2 / s1 + f[1] ** 2 / s2)], ondalik=3)


def _s_det(r):
    a, c = int(r.integers(1, 9)), int(r.integers(1, 9))
    b = int(r.integers(-2, 3))
    while a * c - b * b == 0:
        a += 1
    return soru_sayi(f"S = [[{a}, {b}], [{b}, {c}]] matrisinin determinantı kaçtır?", a * c - b * b,
                     [_md(f"det = {a}×{c} − ({b})² = **{a * c - b * b}**")], [a * c + b * b, a + c, a * c], ondalik=0)


def _s_upgma_ilk(r):
    D = _rastgele_D(r, 4)
    ad = list("ABCD")
    adim, b = hiyerarsik(D, ad, "upgma")
    a = adim[0]
    hedef, f = a["formul"][0]
    v = float(adim[1]["D"][0, 1])
    yanlis = [min(D[a["i"], ad.index(hedef)], D[a["j"], ad.index(hedef)]), max(D[a["i"], ad.index(hedef)], D[a["j"], ad.index(hedef)]),
              float(a["deg"] if "deg" in a else a["deger"])]
    return soru_sayi("Uzaklık matrisi:\n\n" + _md_matris(D, ad) + f"\nUPGMA'nın ilk adımında {a['yeni']} oluşur. d({a['yeni']}, {hedef}) kaçtır?",
                     v, [_md(f"En küçük d = {_tr(a['deger'], 3)} → {a['yeni']}. d({a['yeni']}, {hedef}) = {f}")], yanlis, ondalik=3)


def _s_baglanti(r):
    D = _rastgele_D(r, 4)
    ad = list("ABCD")
    y = _sec_bir(r, ["tek", "tam"])
    adim, _ = hiyerarsik(D, ad, y)
    a = adim[0]
    hedef, f = a["formul"][-1]
    dik, djk = D[a["i"], ad.index(hedef)], D[a["j"], ad.index(hedef)]
    dog = min(dik, djk) if y == "tek" else max(dik, djk)
    yan = [max(dik, djk) if y == "tek" else min(dik, djk), (dik + djk) / 2, dik + djk]
    return soru_sayi("Uzaklık matrisi:\n\n" + _md_matris(D, ad) + f"\n{YONTEM_AD[y]} ile ilk birleşmeden sonra d({a['yeni']}, {hedef}) kaçtır?",
                     dog, [_md(f"İlk birleşme {a['yeni']} (d = {_tr(a['deger'], 3)}). {f}")], yan, ondalik=3)


def _s_ilk_cift(r):
    D = _rastgele_D(r, 5)
    ad = list("ABCDE")
    adim, _ = hiyerarsik(D, ad, "upgma")
    a = adim[0]
    dog = f"{ad[a['i']]} ve {ad[a['j']]}"
    tum = [f"{ad[i]} ve {ad[j]}" for i in range(5) for j in range(i + 1, 5)]
    return _secim(r, "Uzaklık matrisi:\n\n" + _md_matris(D, ad) + "\nHiyerarşik kümelemede ilk birleşen çift hangisidir?", dog,
                  _diger(r, tum, dog), [f"En küçük değer d({ad[a['i']]}, {ad[a['j']]}) = {_tr(a['deger'], 3)}."])


def _s_kofenetik(r):
    D = _rastgele_D(r, 4)
    ad = list("ABCD")
    _, b = hiyerarsik(D, ad, "upgma")
    C = kofenetik(b, 4)
    i, j = sorted(int(x) for x in r.choice(4, 2, replace=False))
    return soru_sayi("Uzaklık matrisi:\n\n" + _md_matris(D, ad) + f"\nUPGMA ağacında {ad[i]} ile {ad[j]} arasındaki kofenetik uzaklık kaçtır?",
                     float(C[i, j]), [_md("Birleşme düzeyleri: " + ", ".join(_tr(x[2], 3) for x in b) +
                                          f". {ad[i]} ile {ad[j]}'yi ilk birleştiren düzey **{_tr(C[i, j], 3)}**")],
                     [float(D[i, j])] + [x[2] for x in b if abs(x[2] - C[i, j]) > 1e-9], ondalik=3)


def _s_nj_q(r):
    D = _rastgele_D(r, 4)
    ad = list("ABCD")
    i, j = sorted(int(x) for x in r.choice(4, 2, replace=False))
    rs = D.sum(axis=1)
    q = 2 * D[i, j] - rs[i] - rs[j]
    return soru_sayi("Uzaklık matrisi:\n\n" + _md_matris(D, ad) + f"\nNJ'nin ilk adımında Q({ad[i]},{ad[j]}) kaçtır?", float(q),
                     [_md(f"n = 4; r({ad[i]}) = {_tr(rs[i])}, r({ad[j]}) = {_tr(rs[j])}. Q = (4 − 2)×{_tr(D[i, j])} − {_tr(rs[i])} − {_tr(rs[j])} = **{_tr(q)}**")],
                     [float(D[i, j] - rs[i] - rs[j]), float(4 * D[i, j] - rs[i] - rs[j]), float(2 * D[i, j] + rs[i] + rs[j])], ondalik=3)


def _s_nj_dal(r):
    D = _rastgele_D(r, 4)
    ad = list("ABCD")
    adim, _ = komsu_birlestirme(D, ad)
    a = adim[0]
    ai = ad[a["i"]]
    return soru_sayi("Uzaklık matrisi:\n\n" + _md_matris(D, ad) + f"\nNJ'nin ilk adımında {ai} ile {ad[a['j']]} komşu seçilir. {ai}'nin dal uzunluğu kaçtır?",
                     a["di"], [_md(f"δ = d/2 + (rᵢ − rⱼ)/[2(n − 2)] = {_tr(D[a['i'], a['j']])}/2 + ({_tr(a['r'][a['i']])} − {_tr(a['r'][a['j']])})/4 = **{_tr(a['di'], 3)}**")],
                     [D[a["i"], a["j"]] / 2, a["dj"], D[a["i"], a["j"]]], ondalik=3)


def _s_nj_yeni(r):
    D = _rastgele_D(r, 4)
    ad = list("ABCD")
    adim, _ = komsu_birlestirme(D, ad)
    a = adim[0]
    hedef, v = a["yeni"][0]
    ik, jk, ij = D[a["i"], ad.index(hedef)], D[a["j"], ad.index(hedef)], D[a["i"], a["j"]]
    return soru_sayi("Uzaklık matrisi:\n\n" + _md_matris(D, ad) + f"\nNJ'de {ad[a['i']]} ve {ad[a['j']]} birleşip U oluşuyor. d(U, {hedef}) kaçtır?",
                     v, [_md(f"d(U,{hedef}) = ({_tr(ik)} + {_tr(jk)} − {_tr(ij)}) / 2 = **{_tr(v, 3)}**")],
                     [(ik + jk) / 2, min(ik, jk), ik + jk - ij], ondalik=3)


def _s_s2d(r):
    s = int(r.integers(5, 96)) / 100
    if r.random() < 0.5:
        return soru_sayi(f"Benzerlik s = {_tr(s)} ise d = 1 − s kaçtır?", 1 - s, [_md(f"1 − {_tr(s)} = **{_tr(1 - s)}**")], [s, 1 + s, s / 2], ondalik=2)
    return soru_sayi(f"Benzerlik s = {_tr(s)} ise d = √(1 − s) kaçtır?", math.sqrt(1 - s), [_md(f"√{_tr(1 - s)} = **{_tr(math.sqrt(1 - s), 3)}**")],
                     [1 - s, math.sqrt(s), (1 - s) ** 2], ondalik=3)


def _s_ucgen(r):
    ab, bc = int(r.integers(1, 8)), int(r.integers(1, 8))
    ac = int(r.integers(1, 16))
    ok = ac <= ab + bc and ab <= ac + bc and bc <= ab + ac
    dog = "Evet, üçgen eşitsizliği sağlanır" if ok else "Hayır, üçgen eşitsizliği bozulur"
    return _secim(r, f"d(A,B) = {ab}, d(B,C) = {bc}, d(A,C) = {ac}. Bu üç değer metrik bir ölçüden gelmiş olabilir mi?", dog,
                  ["Hayır, üçgen eşitsizliği bozulur" if ok else "Evet, üçgen eşitsizliği sağlanır", "Yalnız simetri bozulur", "Karar için karakter sayısı gerekir"],
                  [f"Her kenar öteki ikisinin toplamından küçük ya da eşit olmalı: {ac} ≤ {ab + bc}? {ab} ≤ {ac + bc}? {bc} ≤ {ab + ac}?"])


def _s_hucre(r):
    n, t = int(r.integers(5, 60)), int(r.integers(10, 120))
    return soru_sayi(f"{n} OTU ve {t} karakterden oluşan bir veri matrisinde kaç karakter durumu (hücre) vardır?", n * t,
                     [_md(f"{n} × {t} = **{n * t}**")], [n + t, n * n, t * t], ondalik=0)


def _s_cift(r):
    n = int(r.integers(4, 40))
    return soru_sayi(f"{n} OTU için uzaklık matrisinin köşegen dışında kaç farklı çift değeri vardır?", n * (n - 1) // 2,
                     [_md(f"n(n − 1)/2 = {n}×{n - 1}/2 = **{n * (n - 1) // 2}**")], [n * (n - 1), n * n, n - 1], ondalik=0)


def gen_nt(r):
    sabit = [_s_hucre(r)]
    oge, rol = _sec_bir(r, OTU_OYUN)
    sabit.append(_secim(r, f"“{oge}” bir veri matrisinde neyi temsil eder?", rol, _diger(r, ["OTU", "Karakter", "Karakter durumu", "Uzaklık"], rol),
                        [f"{oge}: **{rol}**."]))
    kar, tip = _sec_bir(r, KARAKTER_TIP_OYUN)
    sabit.append(_secim(r, f"“{kar}” hangi karakter türüdür?", tip, _diger(r, [t for _, t in KARAKTER_TIP_OYUN], tip), [f"**{tip}**."]))
    return _tamamla(r, sabit, _ek(r, S_NT))


def gen_yakin(r):
    ifade, yorum = _sec_bir(r, YAKIN_OYUN)
    sabit = [_s_s2d(r), _s_ucgen(r),
             _secim(r, f"“{ifade}” ne anlama gelir?", yorum, _diger(r, ["Özdeş", "Çok benzer", "Çok farklı", "Zıt örüntü", "Hiç ortak tür yok"], yorum),
                    ["Önce benzerlik mi uzaklık mı olduğuna bakın."])]
    return _tamamla(r, sabit, _ek(r, S_YAKIN))


def gen_oklid(r):
    sabit = [_s_oklid(r), _s_manhattan(r), _sec_bir(r, [_s_z, _s_aralik])(r)]
    return _tamamla(r, sabit, _ek(r, S_OKLID))


def gen_mahal(r):
    sabit = [_s_mahal(r), _s_det(r)]
    return _tamamla(r, sabit, _ek(r, S_MAHAL))


def gen_ikili(r):
    sabit = [_s_ikili(r, _sec_bir(r, ["J", "SM", "D", "JD"])), _s_abcd(r)]
    return _tamamla(r, sabit, _ek(r, S_IKILI))


def gen_upgma(r):
    sabit = [_s_ilk_cift(r), _sec_bir(r, [_s_upgma_ilk, _s_baglanti, _s_kofenetik])(r)]
    return _tamamla(r, sabit, _ek(r, S_UPGMA))


def gen_nj(r):
    sabit = [_sec_bir(r, [_s_nj_q, _s_nj_dal, _s_nj_yeni])(r)]
    return _tamamla(r, sabit, _ek(r, S_NJ))


def _s_bray(r):
    n = int(r.integers(3, 6))
    x = r.integers(0, 12, n)
    y = r.integers(0, 12, n)
    while x.sum() + y.sum() == 0 or np.all(x == y):
        y = r.integers(0, 12, n)
    bc = float(np.abs(x - y).sum() / (x + y).sum())
    return soru_sayi(f"İki alandaki tür bollukları: Alan 1 = {_vek(x)}, Alan 2 = {_vek(y)}. Bray–Curtis uzaklığı kaçtır?", bc,
                     [_md(f"Σ|x − y| = {int(np.abs(x - y).sum())}; Σ(x + y) = {int((x + y).sum())}; BC = **{_tr(bc, 3)}**")],
                     [1 - bc, float(np.abs(x - y).sum()), oklid(x, y) / float((x + y).sum())], ondalik=3)


def _s_korel(r):
    t = 4
    x = r.integers(1, 10, t).astype(float)
    while np.std(x) == 0:
        x = r.integers(1, 10, t).astype(float)
    k = float(r.choice([2, 3, -1, -2]))
    c = float(r.integers(0, 6))
    y = k * x + c
    rr = pearson_r(x, y)
    return soru_sayi(f"Gen X profili {_vek(x)}, gen Y profili {_vek(y)} (Y = {_tr(k)}·X + {_tr(c)}). Korelasyon uzaklığı 1 − r kaçtır?",
                     1 - rr, [_md(f"Y, X'in doğrusal dönüşümü: r = {_tr(rr)}; 1 − r = **{_tr(1 - rr)}**. Öklid ise {_tr(oklid(x, y), 2)} çıkar.")],
                     [rr, oklid(x, y), 1.0], ondalik=2)


def gen_biyo(r):
    v, o, _ = _sec_bir(r, VERI_OLCU)
    sabit = [_sec_bir(r, [_s_bray, _s_korel])(r), _secim(r, f"“{v}” için hangi ölçü uygundur?", o, _diger(r, [x[1] for x in VERI_OLCU], o), [f"**{o}**."])]
    return _tamamla(r, sabit, _ek(r, S_BIYO))


HESAP_URETICI = [_s_oklid, _s_manhattan, _s_z, _s_aralik, lambda r: _s_ikili(r, _sec_bir(r, ["J", "SM", "D", "JD"])), _s_abcd,
                 _s_mahal, _s_det, _s_upgma_ilk, _s_baglanti, _s_ilk_cift, _s_kofenetik, _s_nj_q, _s_nj_dal, _s_nj_yeni,
                 _s_s2d, _s_cift]


def gen_ornek(r):
    return [HESAP_URETICI[int(i)](r) for i in r.permutation(len(HESAP_URETICI))[:SORU_SAYISI]]


def gen_tekrar(r):
    ureticiler = [gen_nt, gen_yakin, gen_oklid, gen_mahal, gen_ikili, gen_upgma, gen_nj, gen_biyo]
    sorular = []
    for u in ureticiler:
        sorular += u(r)[:2]
    sira = r.permutation(len(sorular))
    return [sorular[i] for i in sira][:SORU_SAYISI]


# ===========================================================================
# Gezinme
# ===========================================================================
SAYFALAR = [
    st.Page(sayfa_giris, title="Giriş", icon="🌳", url_path="giris", default=True),
    st.Page(sayfa_nt, title="1 · Numerik taksonomi ve veri matrisi", url_path="numerik-taksonomi"),
    st.Page(sayfa_yakinlik, title="2 · Uzaklık ve yakınlık", url_path="uzaklik-yakinlik"),
    st.Page(sayfa_oklid, title="3 · Öklid uzaklığı", url_path="oklid"),
    st.Page(sayfa_mahalanobis, title="4 · Mahalanobis uzaklığı", url_path="mahalanobis"),
    st.Page(sayfa_ikili, title="5 · İkili veri: Jaccard, basit eşleşme, Dice", url_path="jaccard"),
    st.Page(sayfa_upgma, title="6 · Uzaklık matrisi, UPGMA ve dendrogram", url_path="upgma-dendrogram"),
    st.Page(sayfa_nj, title="7 · Neighbor-Joining", url_path="neighbor-joining"),
    st.Page(sayfa_biyoveri, title="8 · Biyolojik veriler ve ölçü seçimi", url_path="biyolojik-veriler"),
    st.Page(sayfa_ornekler, title="9 · Çözümlü örnekler", url_path="cozumlu-ornekler"),
    st.Page(sayfa_tekrar, title="10 · Genel tekrar", url_path="genel-tekrar"),
]
gezinme = st.navigation(SAYFALAR)

with st.sidebar:
    tohum = st.number_input("Rastgele tohum (0 = rastgele)", 0, 999999, 0,
                            help="Sınıfta herkes aynı sayıyı girerse aynı soruları görür.")
    if st.session_state.get("tohum") != tohum:
        st.session_state.tohum = tohum
        st.session_state.rng = (np.random.default_rng(tohum) if tohum
                                else np.random.default_rng())
    st.divider()
    st.caption(TELIF)

basliklar = [p.title for p in SAYFALAR]
simdiki = basliklar.index(gezinme.title) if gezinme.title in basliklar else 0
secilen = st.selectbox("📚 Bölüm", basliklar, index=simdiki)
if secilen != gezinme.title:
    st.switch_page(SAYFALAR[basliklar.index(secilen)])

gezinme.run()

st.divider()
c1, _, c3 = st.columns([1, 0.2, 1])
if simdiki > 0:
    c1.page_link(SAYFALAR[simdiki - 1], label=f"◀ {basliklar[simdiki - 1]}")
if simdiki < len(SAYFALAR) - 1:
    c3.page_link(SAYFALAR[simdiki + 1], label=f"{basliklar[simdiki + 1]} ▶")

st.markdown(f'<div style="text-align:center;color:gray;font-size:0.85rem;'
            f'margin-top:24px">{TELIF}</div>', unsafe_allow_html=True)
