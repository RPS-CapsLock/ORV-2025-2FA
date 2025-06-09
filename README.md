===============================================
ORV-2FA – API za Dvofaktorsko Avtentikacijo z Obrazom
===============================================

Ta projekt omogoča varen dostop do aplikacij z uporabo tehnologije prepoznavanja obraza kot drugega faktorja avtentikacije (2FA).

---------------------------------------------------------------------------------
1. Kaj je ORV-2FA?
---------------------------------------------------------------------------------

ORV-2FA je spletni program (API), ki uporabnikom omogoča, da svoj obraz 'naučijo' sistemu (usposobijo model) in ga kasneje preverijo za dostop. To povečuje varnost, saj poleg gesla potrebujete še prepoznan obraz.

---------------------------------------------------------------------------------
2. Kaj potrebujem za uporabo?
---------------------------------------------------------------------------------

- Računalnik z operacijskim sistemom Windows, Linux ali macOS
- Namestitev programskega jezika Python (verzija 3.8 ali novejša)
- Internetna povezava za prenos programa in knjižnic
- Osnovno znanje uporabe ukazne vrstice (terminala)

---------------------------------------------------------------------------------
3. Kako namestim program?
---------------------------------------------------------------------------------

a) Prenesite programsko kodo:

   - Če imate nameščen Git (orodje za prenos kode), v ukazni vrstici vpišite:

     git clone https://github.com/ime-uporabnika/orv-2fa.git

   - Če Git nimate, prenesite ZIP arhiv s spletne strani in ga razpakirajte.

b) Odprite ukazno vrstico in se premaknite v mapo projekta, na primer:

   cd orv-2fa

c) Namestite potrebne knjižnice, ki omogočajo prepoznavanje obraza:

   pip install -r requirements.txt

---------------------------------------------------------------------------------
4. Kako zaženem aplikacijo?
---------------------------------------------------------------------------------

Zaženete jo z ukazom:

   python main.py

Program se zažene in začne poslušati zahteve na naslovu:

   http://127.0.0.1:8000

To je lokalni naslov vašega računalnika (localhost).

---------------------------------------------------------------------------------
5. Kako uporabljam aplikacijo?
---------------------------------------------------------------------------------

Odprite spletni brskalnik in obiščite:

   http://127.0.0.1:8000/docs

Videli boste interaktivno dokumentacijo, kjer lahko:

- Naložite več slik svojega obraza, da se sistem nauči prepoznavati vaš obraz (endpoint `/train`)
- Pošljete novo sliko za preverjanje, če je vaš obraz na njej (endpoint `/verify`)

---------------------------------------------------------------------------------
6. Kaj če želim zagnati aplikacijo v Docker okolju?
---------------------------------------------------------------------------------

Docker omogoča enostaven zagon aplikacij v izoliranem okolju.

a) Zgradite Docker sliko:

   docker build -t orv-2fa .

b) Zaženite:

   docker run -p 8000:8000 orv-2fa

Sedaj bo aplikacija na voljo na istem naslovu kot prej.

---------------------------------------------------------------------------------
7. Pogoste težave in rešitve
---------------------------------------------------------------------------------

- **Napaka:** pip ni prepoznan
  - Rešitev: Preverite, da je Python pravilno nameščen in da je ukaz pip dostopen v ukazni vrstici.

- **Napaka:** Modul ni najden
  - Rešitev: Zaženite `pip install -r requirements.txt` znova.

- **Aplikacija ni dosegljiva na http://127.0.0.1:8000**
  - Rešitev: Preverite, da ste aplikacijo pravilno zagnali z `python main.py`.

---------------------------------------------------------------------------------
8. Kontakt in podpora
---------------------------------------------------------------------------------

Za pomoč se obrnite na razvijalca ali preglejte dodatno dokumentacijo v mapi `docs`.

---

Hvala, ker uporabljate ORV-2FA!

