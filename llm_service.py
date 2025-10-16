from sqlalchemy.orm import Session
from models import News
from database import SessionLocal
import ollama
import json

def analyze_news(
    news_id: int,
    db: Session
):
    try:
        news = db.query(News).filter(News.id == news_id).first()
        if news is None:
            raise Exception("Not found")

        if news.analyzed_news is None:
            prompt = f"""Sen profesyonel bir haber analiz ve doğrulama uzmanısın. Aşağıdaki haberi çok detaylı şekilde analiz et ve sonuçları JSON formatında döndür.

                HABERİN BAŞLIĞI: {news.title}
                HABERİN İÇERİĞİ: {news.content}
                HABERİN KAYNAĞI: {news.source}

                Lütfen aşağıdaki yapıda DETAYLI bir JSON çıktısı ver:

                {{
                "genel_bilgiler": {{
                    "ozet": "Haberin kapsamlı 3-4 cümlelik özeti",
                    "ana_konu": "Haberin ana konusu",
                    "alt_konular": ["Alt konu 1", "Alt konu 2", "Alt konu 3"],
                    "kategori": "Politik/Ekonomi/Teknoloji/Spor/Sağlık/Eğitim/Kültür-Sanat/Güvenlik/Çevre/Diğer",
                    "alt_kategoriler": ["Kategori detayı 1", "Kategori detayı 2"]
                }},
                
                "icerik_analizi": {{
                    "haber_turu": "Haber/Röportaj/Analiz/Yorum/Duyuru/Söyleşi",
                    "kaynak_turu": "Ana Haber/İkincil Kaynak/Üçüncü Taraf/Sosyal Medya",
                    "kaynak_guvenilirlik": {{
                    "puan": 0-10 arası sayı,
                    "aciklama": "Kaynağın güvenilirliği hakkında detaylı değerlendirme",
                    }},
                    "objektiflik": {{
                    "puan": 0-10 arası (10 tamamen objektif),
                    "yanlilik_yonu": "Sol/Sağ/Merkez/Yok",
                    "tarafli_ifadeler": ["Örnek ifade 1", "Örnek ifade 2"],
                    "degerlendirme": "Detaylı objektiflik analizi"
                    }},
                    "dil_ve_uslup": {{
                    "ton": "Resmi/Gayri Resmi/Teknik/Popüler/Sensasyonel",
                    "dil_seviyesi": "Basit/Orta/Karmaşık/Teknik",
                    "retorik_unsurlar": ["Metafor kullanımı", "Abartı", "vs."],
                    "manipulatif_dil": "Var/Yok - açıklama"
                    }}
                }},
                
                "duygu_ve_ton_analizi": {{
                    "genel_ton": "Pozitif/Negatif/Nötr/Karışık",
                    "duygu_skoru": 0-10 arası sayı (-5 ile +5 arası da olabilir),
                    "duygusal_yogunluk": "Düşük/Orta/Yüksek",
                    "baskın_duygular": ["Öfke", "Umut", "Korku", "Şaşkınlık", "vs."],
                    "hedef_kitle_etkisi": "Haberin hedef kitlede yaratacağı duygusal etki",
                    "polarizasyon_potansiyeli": {{
                    "seviye": "Düşük/Orta/Yüksek",
                    "aciklama": "Toplumsal kutuplaşma yaratma potansiyeli"
                    }}
                }},
                
                "icerik_unsurlari": {{
                    "anahtar_kelimeler": ["En önemli 10 kelime"],
                    "onemli_kisiler": [
                    {{"isim": "Kişi adı", "rol": "Pozisyon/Rolü", "bahsedilme_baglamı": "Nasıl bahsedildi"}}
                    ],
                    "onemli_kurumlar": [
                    {{"isim": "Kurum adı", "tur": "Kamu/Özel/STK/vs.", "rol": "Haberdeki rolü"}}
                    ],
                    "tarihler_ve_olaylar": [
                    {{"tarih": "Tarih bilgisi", "olay": "Ne oldu", "onem": "Neden önemli"}}
                    ],
                    "sayisal_veriler": [
                    {{"veri": "İstatistik/Sayı", "baglamı": "Ne anlama geliyor", "kaynak": "Kaynak belirtilmiş mi"}}
                    ],
                    "yer_bilgisi": [
                    {{"konum": "Şehir/Ülke", "onemi": "Haberdeki önemi", "etki_alani": "Yerel/Bölgesel/Küresel"}}
                    ],
                    "alintilar": [
                    {{"alinti": "Doğrudan alıntı", "kaynak": "Kim söyledi", "baglamı": "Neden önemli"}}
                    ]
                }},
                
                "dogruluk_ve_guvenilirlik": {{
                    "genel_guvenilirlik": {{
                    "puan": 0-10 arası sayı,
                    "degerlendirme": "Kapsamlı güvenilirlik değerlendirmesi"
                    }},
                    "dogrulanabilirlik": {{
                    "seviye": "Yüksek/Orta/Düşük",
                    "dogrulanabilir_iddialar": ["İddia 1", "İddia 2"],
                    "dogrulanamayan_iddialar": ["İddia 1", "İddia 2"],
                    "eksik_bilgiler": ["Hangi bilgiler eksik"]
                    }},
                    "yaniltici_icerik_riski": {{
                    "risk_seviyesi": "Düşük/Orta/Yüksek/Kritik",
                    "potansiyel_yanlis_bilgiler": ["Şüpheli bilgi 1", "Şüpheli bilgi 2"],
                    "clickbait_unsurlar": "Var/Yok - açıklama",
                    "manşet_icerik_uyumu": "Yüksek/Orta/Düşük - açıklama",
                    "baglamsal_yaniltma": "Var mı, nasıl?"
                    }},
                    "fact_check_onerileri": [
                    "Kontrol edilmesi gereken iddia 1",
                    "Kontrol edilmesi gereken iddia 2"
                    ]
                }},
                
                "etki_ve_kapsam_analizi": {{
                    "toplumsal_etki": {{
                    "seviye": "Düşük/Orta/Yüksek/Kritik",
                    "etkilenecek_gruplar": ["Grup 1", "Grup 2"],
                    "kisa_vadeli_etkiler": ["Etki 1", "Etki 2"],
                    "uzun_vadeli_etkiler": ["Etki 1", "Etki 2"]
                    }},
                    "ekonomik_etki": {{
                    "var_mi": true/false,
                    "seviye": "Düşük/Orta/Yüksek",
                    "etkilenen_sektorler": ["Sektör 1", "Sektör 2"],
                    "aciklama": "Ekonomik etki detayı"
                    }},
                    "politik_etki": {{
                    "var_mi": true/false,
                    "seviye": "Düşük/Orta/Yüksek",
                    "etkilenen_taraflar": ["Parti/Grup 1", "Parti/Grup 2"],
                    "aciklama": "Politik etki detayı"
                    }},
                    "aciliyet_durumu": {{
                    "seviye": "Düşük/Orta/Yüksek/Acil/Kriz",
                    "zaman_duyarliligi": "Haberin zaman açısından aciliyeti",
                    "eylem_gerektiriyor_mu": "Evet/Hayır - açıklama"
                    }},
                    "kapsam": {{
                    "cografik": "Yerel/Bölgesel/Ulusal/Uluslararası/Küresel",
                    "demografik": ["Hangi yaş/cinsiyet/grup etkileniyor"],
                    "sektorel": ["Hangi sektörler etkileniyor"]
                    }}
                }},
                
                "paydas_analizi": {{
                    "kazanan_taraflar": [
                    {{
                        "grup": "Grup/Kurum/Kişi",
                        "kazanim": "Ne kazanıyor",
                        "etki_derecesi": "Düşük/Orta/Yüksek"
                    }}
                    ],
                    "kaybeden_taraflar": [
                    {{
                        "grup": "Grup/Kurum/Kişi",
                        "kayip": "Ne kaybediyor",
                        "etki_derecesi": "Düşük/Orta/Yüksek"
                    }}
                    ],
                    "notr_taraflar": ["Etkilenmeyen veya nötr kalan gruplar"],
                    "cikis_catismasi": {{
                    "var_mi": true/false,
                    "taraflar": ["Taraf 1 vs Taraf 2"],
                    "catisma_konusu": "Ne üzerine çatışma var"
                    }}
                }},
                
                "baglamsal_analiz": {{
                    "tarihsel_baglam": "Bu haberin tarihsel arka planı",
                    "onceki_gelismeler": ["Bu habere götüren önceki olaylar"],
                    "trend_analizi": "Bu haber bir trendin parçası mı?",
                    "genis_resim": "Bu haber büyük resimde nereye oturuyor"
                }},
                
                "eksik_yonler_ve_sorular": {{
                    "cevaplanmayan_sorular": [
                    "Kim? Ne? Nerede? Ne zaman? Neden? Nasıl? sorularından hangisi eksik"
                    ],
                    "eksik_perspektifler": ["Hangi bakış açıları eksik"],
                    "gerekli_ek_bilgiler": ["Tam anlamak için hangi bilgiler lazım"],
                    "karsi_gorusler": "Karşı görüşler dengeli şekilde sunulmuş mu?"
                }},
                
                "okuyucu_icin_degerler": {{
                    "guvenilirlik_skoru": 0-100 arası,
                    "onem_derecesi": "Düşük/Orta/Yüksek/Kritik",
                    "okuma_onceligi": "Düşük/Orta/Yüksek/Acil",
                    "ilgi_alanlari": ["Teknoloji meraklıları", "Politika takipçileri", "vs."],
                    "eyleme_donusturulebilirlik": {{
                    "seviye": "Düşük/Orta/Yüksek",
                    "uygulanabilir_adimlar": ["Okuyucunun yapabileceği şeyler"]
                    }}
                }},
                
                "oneriler_ve_uyarilar": {{
                    "okuyucu_onerileri": [
                    "Okuyucuya kritik öneri 1",
                    "Okuyucuya kritik öneri 2"
                    ],
                    "dikkat_edilmesi_gerekenler": [
                    "Bu haberi okurken nelere dikkat edilmeli"
                    ],
                    "ek_arastirma_onerileri": [
                    "Hangi konular daha fazla araştırılmalı"
                    ],
                    "ilgili_kaynaklar": [
                    "Başvurulabilecek güvenilir kaynaklar"
                    ]
                }},
                
                "meta_analiz": {{
                    "haberin_amaci": "Bu haberin yayınlanma amacı nedir?",
                    "hedef_kitle": "Haberin hedef kitlesi kimler?",
                    "zamanlamasi": "Haberin zamanlaması önemli mi?",
                    "sosyal_medya_potansiyeli": {{
                    "virallik": "Düşük/Orta/Yüksek",
                    "paylasim_motivasyonu": "İnsanlar neden paylaşır",
                    "tartisma_potansiyeli": "Düşük/Orta/Yüksek"
                    }},
                    "etik_degerlendirme": {{
                    "etik_sorunlar": "Var/Yok - açıklama",
                    "mahremiyet_ihlali": "Var/Yok - açıklama",
                    "zarar_verme_potansiyeli": "Düşük/Orta/Yüksek"
                    }}
                }},
                
                "gelecek_projeksiyonları": {{
                    "kisa_vade": "1 hafta içinde ne olabilir",
                    "orta_vade": "1-3 ay içinde ne olabilir",
                    "uzun_vade": "6+ ay içinde ne olabilir",
                    "takip_edilmesi_gerekenler": ["İzlenmesi gereken gelişmeler"]
                }},
                
                "genel_degerlendirme": {{
                    "guclu_yonler": ["Haberin güçlü yönleri"],
                    "zayif_yonler": ["Haberin zayıf yönleri"],
                    "genel_kalite_puani": 0-100 arası,
                    "son_yorum": "Haberin genel değerlendirmesi (2-3 cümle)"
                }}
                }}

                ÖNEMLI KURALLAR:
                1. Sadece JSON formatında yanıt ver, başka açıklama ekleme
                2. Tüm alanları doldur, "Bilgi yok" yerine mantıklı tahminler yap
                3. Sayısal puanları 0-10 veya 0-100 ölçeğinde ver
                4. Boolean değerler için true/false kullan
                5. Boş array yerine en azından genel değerlendirme ekle
                6. Türkçe karakterleri doğru kullan
            """
    
            response = ollama.chat(
            model="gpt-oss:120b-cloud",
            messages=[{"role": "user", "content": prompt}],
            format="json"  # JSON formatını zorunlu kılar
            )

            try:
                # JSON'u parse et
                analiz = json.loads(response["message"]["content"])
                news.analyzed_news = analiz
                db.commit()
                db.refresh(news)
            except json.JSONDecodeError as e:
                print("JSON parse hatası:", e)
                print("Ham yanıt:", response["message"]["content"])
                raise Exception("JSON PARSE ERROR")
            return analiz
        else:
            return news.analyzed_news
    except Exception as e:
        print(e)
        raise e
    finally:
        db.close()


# analiz_result = analyze_news(1041)
# print(analiz_result)