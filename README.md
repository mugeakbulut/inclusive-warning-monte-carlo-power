# WP3 Monte Carlo Mediation Power Analysis

Bu depo, TÜBİTAK 1001 proje önerisinin WP3 iş paketi için yapılan 2×2×2 tekrarlı ölçümlü aracılık gücü simülasyonunu içerir. 

## Araştırma modeli

Birinci doğrulayıcı dolaylı etki `sade dil → algılanan güvenilirlik → davranışsal niyet` yoludur. İkinci doğrulayıcı dolaylı etki `somut eylem yönlendirmesi → algılanan güvenilirlik → davranışsal niyet` yoludur. Analiz toplam örneklem için planlanmıştır. Hedef gruplar arasındaki farklılıklar keşifsel bırakılmıştır.

Tasarım dört eşit hedef grup, 288 tamamlanmış katılımcı ve kişi başına sekiz mesaj içerir. Dört temel senaryo, sekiz deney koşuluna dengeli biçimde atanır. Sunum sırası dengeli Latin kareyle düzenlenir.

## Ana sonuç

| Temkinli senaryo, N=288 | α=.05 | α=.025 |
|---|---:|---:|
| Sade dil dolaylı etkisi | %91,8 | %87,4 |
| Somut eylem dolaylı etkisi | %93,1 | %88,8 |

Mevcut 288 tamamlanmış katılımcı ve yüzde 15 kayıp payıyla 340 kişilik başlangıç hedefi korunabilir. Ana koşullar 5.000 kez üretilmiştir. Orta ve iyimser senaryolarda güç yüzde 99'un üzerindedir. Etki değerleri pilot bulgu değildir. Bunlar planlama amacıyla belirlenen standartlaştırılmış varsayımlardır.

Üç deneysel değişkenin düzeyleri −0,5 ve +0,5 olarak kodlanmıştır. Güç, dolaylı etkiye ait Monte Carlo güven aralığının sıfırı içermediği tekrarların oranıdır.

## Depo yapısı

`src/wp3_mediation_power.py` simülasyon kodudur.

`tests/test_smoke.py` tasarım ve temel çalışma kontrollerini içerir.

`results/` ana sonuçları, örneklem taramasını ve duyarlılık kontrollerini içerir.

`docs/Monte_Carlo_Aracilik_Guc_Analizi_Raporu.docx` kısa teknik rapordur.

`docs/index.html` GitHub Pages veya kişisel alan adı için hazırlanmış sonuç sayfasıdır.

Başvuru formu kamuya açık depoya yanlışlıkla yüklenmemesi için bu pakete eklenmemiştir.

## Kurulum

Python 3.10 veya üzeri kullanılmalıdır.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows üzerinde etkinleştirme komutu şöyledir.

```powershell
.venv\Scripts\activate
```

## Simülasyonu çalıştırma

Tam analiz aşağıdaki komutla yeniden üretilebilir.

```bash
python src/wp3_mediation_power.py --output-dir results/reproduction
```

Yalnızca N=288 ana koşullarını 5.000 tekrarla yeniden üretmek için şu komut kullanılabilir.

```bash
python scripts/run_main_5000.py
```

Kısa teknik kontrol için şu komut kullanılabilir.

```bash
python src/wp3_mediation_power.py --quick --output-dir results/quick-check
```

Ana rastgele sayı tohumu `20260820` olarak sabitlenmiştir.

## Test

```bash
python -m pip install -r requirements-dev.txt
pytest -q
```

## GitHub Pages

Depo ayarlarında `Pages` bölümünü açın. Kaynak olarak ana dalın `/docs` klasörünü seçin. `docs/index.html` sonuç sayfası olarak yayımlanacaktır.

## Veri notu

Depoda gerçek katılımcı verisi yoktur. Kod yalnızca sentetik veri üretir. Güvenilirlik ve davranışsal niyet, sürekli madde ortalaması puanları olarak modellenmiştir.

