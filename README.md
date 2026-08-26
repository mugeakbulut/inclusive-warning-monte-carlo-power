# Inclusive Warning Monte Carlo Power

Bu depo, TÜBİTAK 1001 proje önerisinin WP3 iş paketi için yapılan 2×2×2 tekrarlı ölçümlü aracılık gücü simülasyonunu içerir.

## Model

Doğrulayıcı dolaylı etkiler şunlardır:

1. `sade dil → algılanan güvenilirlik → davranışsal niyet` (`a₁×b`)
2. `somut eylem yönlendirmesi → algılanan güvenilirlik → davranışsal niyet` (`a₂×b`)

Analiz 288 tamamlanmış katılımcıdan oluşan toplam örneklem için planlanmıştır. Dört hedef grup eşittir (72 kişi/grup); grup farklılıkları keşifseldir. Her katılımcı sekiz koşulun tamamını değerlendirir.

Birincil model katılımcı rastgele kesişimi ile sadelik, eylem yönlendirmesi ve odak için üç birbiriyle ilişkisiz rastgele eğim içerir. Dört temel senaryo sabit etki, deneysel değişkenler −0,5/+0,5 kodlu sabit etkiler olarak modele girer.

## Ana sonuç

| Temkinli senaryo, N=288 | α=.05 | α=.025 |
|---|---:|---:|
| Sade dil dolaylı etkisi | %89,7 | %84,3 |
| Somut eylem dolaylı etkisi | %90,3 | %84,5 |

Ana koşulların her biri 5.000 kez üretilmiştir. Bonferroni düzeltmeli birincil ölçütte (`α=.025`) iki dolaylı etkinin ayrı ayrı gücü %80'in üzerindedir. Orta ve iyimser senaryolarda güç %99'un üzerindedir. Bu nedenle 288 tamamlanmış katılımcı ve %15 kayıp payıyla 340 kişilik başlangıç hedefi korunabilir.

Etki büyüklükleri ve varyans bileşenleri pilot bulgu değil, standartlaştırılmış planlama varsayımlarıdır. Güç, dolaylı etkiye ait Monte Carlo ürün güven aralığının sıfırı içermediği tekrarların oranıdır.

## Tekillik ve yakınsamama kuralı

Nihai veri analizinde model tekil sonuç verir veya yakınsamazsa tahmini varyansı en küçük olan rastgele eğim çıkarılır ve model yeniden kurulur. Sorun sürerse aynı işlem tekrarlanır. Varyansların tam eşit olması halinde çıkarma sırası odak, eylem yönlendirmesi ve sadelik biçimindedir. Katılımcı rastgele kesişimi her durumda korunur.

## Dosyalar

- `src/wp3_mediation_power.py`: Tam simülasyon kodu
- `scripts/run_main_5000.py`: N=288 ana koşullarının 5.000 tekrarlı koşusu
- `results/`: Ana, örneklem taraması, duyarlılık ve yanlış-pozitif sonuçları
- `docs/Monte_Carlo_Aracilik_Guc_Analizi_Raporu.docx`: Teknik rapor
- `docs/BASVURUYA_EKLENECEK_METIN.txt`: Başvuru metni için kısa ek
- `tests/test_smoke.py`: Tasarım ve çalışma kontrolleri

Başvuru formu kamuya açık depoya yanlışlıkla yüklenmemesi için pakete eklenmemiştir.

## Kurulum ve çalıştırma

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/wp3_mediation_power.py --workers 8 --output-dir results
```

Yalnızca ana 5.000 tekrarlı koşu:

```bash
python scripts/run_main_5000.py
```

Kısa kod kontrolü:

```bash
python src/wp3_mediation_power.py --quick --workers 4 --output-dir results/quick-check
```

Ana rastgele sayı tohumu `20260826` olarak sabitlenmiştir. Depoda gerçek katılımcı verisi yoktur; yalnızca sentetik veri üretilir.

## Lisans

Bu pakete lisans atanmadı. Depoyu kamuya açmadan önce ekip uygun lisansı belirlemelidir.
