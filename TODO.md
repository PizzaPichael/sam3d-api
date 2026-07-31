# TODO

## Maske aus Alpha-Kanal serverseitig ableiten

Aktuell macht das nur `client/generate3d.py` (`mask_from_alpha`). Besser waere es in
`api.py` in `/generate-3d`: `mask` optional machen, und wenn sie fehlt, aus dem
Alpha-Kanal des uebergebenen Bildes ableiten.

Vorteil: gilt fuer alle Clients, auch die Unity/MR-App — die muss dann keine eigene
Alpha-Extraktion implementieren. Der Client braeuchte dann auch kein Pillow mehr.

Umsetzung:
- `Generate3dRequest.mask` auf `Optional[str] = None`
- fehlt `mask`: Bild mit PIL oeffnen, `"A" in image.getbands()` pruefen, Alpha > 0 als
  Maske; sonst 400 mit klarer Meldung ("kein Alpha-Kanal, bitte Maske mitschicken")
- danach `mask_from_alpha` im Client entfernen und Fallback auf den Server-Pfad

Nach dem Umbau: Pod braucht `git pull` + API-Neustart.
