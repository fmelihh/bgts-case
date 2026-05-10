## **KB-01: BGP Troubleshooting ve RCA Kılavuzu** 

_Versiyon: 2.3 | Son Güncelleme: 2024-01-15 | Ekip: Network Operasyon_ 

## **1. BGP Durum Makinesi** 

BGP oturumu aşağıdaki durumlar üzerinden geçer. Her geçiş bir event tetikler: 

|**Durum**|**Açıklama**|**Beklenen Süre**|**Sorun Sinyali**|
|---|---|---|---|
|Idle|Başlangıç durumu, bağlantı bekleniyor|< 5 sn|Sürekli Idle kalıyorsa ACL/route problemi|
|Connect|TCP bağlantısı kuruluyor|< 10 sn|Timeout → firewall bloğu|
|Active|TCP başarısız, yeniden deneniyor|< 30 sn|Loop → neighbor IP yanlış|
|OpenSent|OPEN mesajı gönderildi|< 5 sn|Uzun kalıyorsa AS numarası uyuşmuyor|
|OpenConfirm|KEEPALIVE bekleniyor|< 5 sn|Hold timer mismatch|
|Established|Oturum aktif, prefix alışverişi|Sürekli|Düşerse → log analizi şart|



## **2. Yaygın BGP Hata Logları ve Anlamları** 

```
%BGP-5-ADJCHANGE: neighbor 195.142.11.1 Down BGP Notification sent
--> Notification sent: bizim tarafimiz oturumu kapatti. Neden: hold timer expired.
```

```
%BGP-3-NOTIFICATION: sent to neighbor 10.0.0.1 4/0 (hold time expired) 0 bytes
--> Keepalive paketleri karsi tarafa ulasmıyor. MTU sorunu veya QoS drop.
```

```
%BGP-5-ADJCHANGE: neighbor 203.0.113.5 Down Peer closing down the session
--> Karsi taraf (ISP) oturumu kapatti. ISP NOC ile iletisime gecilmeli.
%BGP-3-MAXPFXEXCEED: No. of prefix received from 10.10.10.1 reaches 500
--> maximum-prefix limiti asildi. Route leak veya yanlis politika.
```

```
BGP: 195.142.11.1 went from Established to Idle
OSPF: Route 0.0.0.0/0 removed from routing table
--> BGP dustugunde default route da kalkiyor. Floating static route tanimlanmamis.
```

## **3. Tanılama Komutları (Cisco IOS/IOS-XE)** 

|**Komut**|**Amaç**|**Kritik Çıktı**|
|---|---|---|
|show bgp summary|Tüm neighbor durumu|State sütunu: Established?|
|show bgp neighbors 195.x.x.x|Detaylı neighbor bilgisi|BGP state, hold time, prefix count|
|show bgp neighbors X advertised-routes|Gönderilen prefixler|Route policy doğru çalışıyor mu?|
|show bgp neighbors X received-routes|Alınan prefixler|route-map in uygulandı mı?|
|debug ip bgp 195.x.x.x events|Canlı event izleme|SADECE üretimde dikkatli kullan|
|clear ip bgp 195.x.x.x soft|Soft reset (trafiği kesmez)|Policy değişikliği sonrası|



## **4. RCA Karar Ağacı** 

**Adım 1:** BGP düştü mü? → show bgp summary ile kontrol et 

**Adım 2:** Notification mesajı var mı? → Yukarıdaki log tablosuna bak **Adım 3:** Hold timer expire mı? → MTU/QoS kontrol et (ping df-bit ile) **Adım 4:** ISP kaynaklı mı? → traceroute + ISP NOC ticket aç 

**Adım 5:** Route leak var mı? → prefix-list ve route-map gözden geçir 

## **5. Önleyici Tedbirler** 

- BFD (Bidirectional Forwarding Detection) etkinleştir: hold timer 3 sn altına iner 

- BGP community ile route tagging yap, failover politikasını netleştir 

- maximum-prefix limiti ISP ile mutabık kalınan değerin %80'ine ayarla 

- Değişiklik öncesi 'show bgp summary' çıktısını kaydet (baseline) 

