===============================================
ORV-2FA – API za Dvofaktorsko Avtentikacijo z Obrazom
===============================================

Ta projekt omogoča varen dostop do aplikacij z uporabo tehnologije prepoznavanja obraza kot drugega faktorja avtentikacije (2FA).

---------------------------------------------------------------------------------
1. Osnovni koncept
---------------------------------------------------------------------------------

ORV-2FA omogoča, da vaš obraz postane vaš drugi faktor za prijavo (poleg gesla). To pomeni, da poleg klasične prijave s uporabniškim imenom in geslom, sistem preveri tudi vaš obraz.

To poteka tako, da:

- Najprej naučite (usposobite) model na podlagi vaših slik obraza.
- Kasneje, ko se želite prijaviti, aplikacija posreduje vaš obraz API-ju, ki preveri, ali se ujema s shranjenim modelom.
- Če se ujemata, vas sistem pusti naprej (uspešna prijava).

---------------------------------------------------------------------------------
2. Kako aplikacija pošlje podatke API-ju?
---------------------------------------------------------------------------------

Vaša aplikacija (lahko spletna stran, mobilna aplikacija ali namizna aplikacija) komunicira z ORV-2FA API-jem preko HTTP zahtevkov (to so kot pošiljanje sporočil prek interneta).

a) Treniranje modela (usposabljanje)  
- Ko uporabnik želi "naučiti" sistem svoj obraz, aplikacija zbere več slik obraza (npr. slikanje preko kamere).  
- Te slike se pošljejo API-ju z zahtevkom POST na `/train` endpoint.  
- API sprejme slike, jih obdela in ustvari model, ki shrani značilnosti obraza tega uporabnika.

b) Preverjanje obraza (verifikacija)  
- Ob prijavi aplikacija zajame trenutno sliko uporabnikovega obraza.  
- Sliko pošlje API-ju z zahtevkom POST na `/verify` endpoint skupaj z identifikatorjem uporabnika.  
- API primerja trenutno sliko z usposobljenim modelom in vrne odgovor, ali se obraz ujema ali ne.  
- Če je ujemanje uspešno, aplikacija dovoli dostop.

---------------------------------------------------------------------------------
3. Primer poteka komunikacije med aplikacijo in ORV-2FA API:
---------------------------------------------------------------------------------

| Korak           | Kaj se zgodi                              | Komunikacija                      |
|-----------------|------------------------------------------|----------------------------------|
| 1. Zajem slik   | Aplikacija posname več slik uporabnikovega obraza | Lokalno na napravi               |
| 2. Pošiljanje slik | Slike se pošljejo API-ju za trening      | POST /train z več slikami         |
| 3. Shranjevanje modela | API ustvari model obraza za uporabnika  | Na strežniku                     |
| 4. Prijava uporabnika | Aplikacija zajame obraz in ga pošlje na preverbo | POST /verify z eno sliko in ID uporabnika |
| 5. Preverjanje  | API primerja obraz z modelom in vrne rezultat | Odgovor API-ja: uspeh ali neuspeh |
| 6. Dostop       | Aplikacija omogoči ali zavrne dostop     | Lokalno na napravi                |

---------------------------------------------------------------------------------
4. Podrobnejši opis API klicev
---------------------------------------------------------------------------------

a) POST `/train`  
- Podatki: več slik obraza (npr. base64 kodirane ali kot datoteke)  
- Namen: ustvariti model za uporabnika  
- Vhod: ID uporabnika, slike obraza  
- Izhod: status uspeha, sporočilo o uspehu ali napaki

b) POST `/verify`  
- Podatki: ena slika obraza za preverjanje, ID uporabnika  
- Namen: preveriti, ali se obraz ujema s shranjenim modelom  
- Izhod: rezultat preverjanja (true/false), odstotek ujemanja, napaka če ni modela

---------------------------------------------------------------------------------
5. Kaj se dogaja "pod pokrovom"?
---------------------------------------------------------------------------------

- **Predprocesiranje slik:** slike se izboljšajo, normalizirajo in povečajo (augmentation) za boljšo robustnost.  
- **Globoko učenje:** uporabljamo model MobileNetV2, ki je nevronska mreža, prilagojena za prepoznavanje obraznih značilnosti.  
- **Shranjevanje modelov:** vsak uporabnik ima svoj model, ki omogoča hitro in natančno preverjanje.  
- **Dostopnost:** API je RESTful, zato ga lahko kliče katerakoli aplikacija, ki zna pošiljati HTTP zahtevke.

---------------------------------------------------------------------------------
6. Kaj potrebujem za uporabo?
---------------------------------------------------------------------------------

- Računalnik z operacijskim sistemom Windows, Linux ali macOS  
- Namestitev programskega jezika Python (verzija 3.8 ali novejša)  
- Internetna povezava za prenos programa in knjižnic  
- Osnovno znanje uporabe ukazne vrstice (terminala)

---------------------------------------------------------------------------------
7. Kako namestim program?
---------------------------------------------------------------------------------

a) Prenesite programsko kodo:

- Če imate nameščen Git, v ukazni vrstici vpišite:  
  `git clone https://github.com/ime-uporabnika/orv-2fa.git`

- Če Git nimate, prenesite ZIP arhiv s spletne strani in ga razpakirajte.

b) Odprite ukazno vrstico in se premaknite v mapo projekta:  
`cd orv-2fa`

c) Namestite potrebne knjižnice:  
`pip install -r requirements.txt`

---------------------------------------------------------------------------------
8. Kako zaženem aplikacijo?
---------------------------------------------------------------------------------

Zaženete jo z ukazom:  
`python API.py`

Program začne poslušati zahteve na naslovu:  
`http://127.0.0.1:8000`

---------------------------------------------------------------------------------
9. Pogoste težave in rešitve
---------------------------------------------------------------------------------

- **Napaka:** pip ni prepoznan  
  - Rešitev: Preverite, da je Python pravilno nameščen in da je ukaz pip dostopen v ukazni vrstici.

- **Napaka:** Modul ni najden  
  - Rešitev: Zaženite `pip install -r requirements.txt` znova.

- **Aplikacija ni dosegljiva na http://127.0.0.1:8000**  
  - Rešitev: Preverite, da ste aplikacijo pravilno zagnali z `python API.py`.

---------------------------------------------------------------------------------
10. Zagon v Docker okolju (neobvezno)
---------------------------------------------------------------------------------

Če imate nameščen Docker, lahko aplikacijo zaženete tudi tako:

a) Zgradite Docker sliko:  
`docker build -t orv-2fa .`

b) Zaženite:  
`docker run -p 8000:8000 orv-2fa`

Aplikacija bo dostopna na istem naslovu kot prej.

