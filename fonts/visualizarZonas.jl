using Images
using FileIO
using Statistics
using Plots
using ImageBinarization
using ImageMorphology

# --- CONFIGURACIÓN ---
ruta_img = "DatasetLogos/train_filtrado/adidas_fake/tadidas_175.jpg"

if !isfile(ruta_img)
    error("No se encuentra la imagen en: $ruta_img")
end

# =============================================================================
# 1. CARGAR Y PREPROCESAR
# =============================================================================
img     = FileIO.load(ruta_img)
grayImg = Gray.(img)
rows, cols = size(grayImg)

# --- Selección automática de máscara: logo claro sobre oscuro o viceversa ---
umbral_otsu    = otsu_threshold(grayImg)
mascara_oscura = grayImg .< umbral_otsu
mascara_clara  = grayImg .> umbral_otsu

# Score: premia máscaras con pocos componentes grandes (logo bien unido)
function score_mascara(mascara)
    etiq  = label_components(closing(mascara, trues(3,3)))
    n_lbl = maximum(etiq)
    n_lbl < 2 && return 0.0
    areas = [sum(etiq .== lbl) for lbl in 2:n_lbl]
    return Float64(maximum(areas)^2) / length(areas)
end

if score_mascara(mascara_oscura) > score_mascara(mascara_clara)
    mascara_inicial = mascara_oscura
    println("Logo OSCURO sobre fondo claro.")
else
    mascara_inicial = mascara_clara
    println("Logo CLARO sobre fondo oscuro.")
end

# =============================================================================
# 2. LIMPIEZA MORFOLÓGICA LEVE
#    Cierre PEQUEÑO (3x3) para no fusionar símbolo y texto.
# =============================================================================
mascara = closing(mascara_inicial, trues(3, 3))

# Eliminar ruido mínimo por componentes
area_min_px    = round(Int, rows * cols * 0.0005)
etiq_raw       = label_components(mascara)
mascara_limpia = falses(rows, cols)
for lbl in 1:maximum(etiq_raw)
    if sum(etiq_raw .== lbl) >= area_min_px
        mascara_limpia .|= (etiq_raw .== lbl)
    end
end
mascara = mascara_limpia

# =============================================================================
# 3. BOUNDING BOX GLOBAL DEL LOGO
# =============================================================================
filas_activas = findall(vec(any(mascara, dims=2)))
cols_activas  = findall(vec(any(mascara, dims=1)))

if isempty(filas_activas) || isempty(cols_activas)
    error("No se detectó contenido en la imagen.")
end

r_global_min = first(filas_activas)
r_global_max = last(filas_activas)
c_global_min = first(cols_activas)
c_global_max = last(cols_activas)

# =============================================================================
# 4. PERFIL DE PROYECCIÓN HORIZONTAL
#    Cuenta píxeles activos por fila. El hueco entre símbolo y texto
#    aparece como un mínimo local en este perfil.
# =============================================================================
region_logo = mascara[r_global_min:r_global_max, c_global_min:c_global_max]
perfil      = Float64.(vec(sum(region_logo, dims=2)))
n_filas     = length(perfil)

# Suavizar para eliminar falsos mínimos
ventana = max(3, round(Int, n_filas * 0.03))
function suavizar(v, w)
    out = zeros(Float64, length(v))
    for i in eachindex(v)
        i0 = max(1, i - w); i1 = min(length(v), i + w)
        out[i] = mean(v[i0:i1])
    end
    return out
end
perfil_suave = suavizar(perfil, ventana)

# Buscar mínimo solo en la zona central para evitar bordes
zona_ini = max(1, round(Int, n_filas * 0.25))
zona_fin = min(n_filas, round(Int, n_filas * 0.75))
_, idx_min_local = findmin(perfil_suave[zona_ini:zona_fin])
fila_corte_rel = zona_ini + idx_min_local - 1        # relativa al bbox global
fila_corte_abs = r_global_min + fila_corte_rel - 1   # absoluta en la imagen

# =============================================================================
# 5. VALIDAR SI HAY HUECO REAL
#    El mínimo debe ser notablemente menor que las medias de arriba y abajo.
# =============================================================================
media_sup   = mean(perfil_suave[1:fila_corte_rel])
media_inf   = mean(perfil_suave[fila_corte_rel:end])
val_min     = perfil_suave[fila_corte_rel]
umbral_sep  = 0.60   # el hueco debe ser < 60% de la media más baja

hay_sep = val_min < umbral_sep * min(media_sup, media_inf)
println("Perfil → mínimo=$(round(val_min,digits=1)), " *
        "media_sup=$(round(media_sup,digits=1)), " *
        "media_inf=$(round(media_inf,digits=1))  [$(hay_sep ? "SEPARACIÓN DETECTADA" : "sin separación")]")

# =============================================================================
# 6. BOUNDING BOXES FINALES
# =============================================================================
margen = 3

function tight_bbox(m, r0, r1, c0, c1)
    sub   = m[r0:r1, c0:c1]
    fr    = findall(vec(any(sub, dims=2)))
    fc    = findall(vec(any(sub, dims=1)))
    (isempty(fr) || isempty(fc)) && return nothing
    ra = clamp(r0 + first(fr)  - 1 - margen, 1, size(m,1))
    rb = clamp(r0 + last(fr)   - 1 + margen, 1, size(m,1))
    ca = clamp(c0 + first(fc)  - 1 - margen, 1, size(m,2))
    cb = clamp(c0 + last(fc)   - 1 + margen, 1, size(m,2))
    return (ra, rb, ca, cb)
end

if hay_sep
    bb_simbolo = tight_bbox(mascara,
                    r_global_min, fila_corte_abs,
                    c_global_min, c_global_max)
    bb_texto   = tight_bbox(mascara,
                    fila_corte_abs, r_global_max,
                    c_global_min, c_global_max)
else
    bb_simbolo = (clamp(r_global_min - margen, 1, rows),
                  clamp(r_global_max + margen, 1, rows),
                  clamp(c_global_min - margen, 1, cols),
                  clamp(c_global_max + margen, 1, cols))
    bb_texto   = nothing
end

# =============================================================================
# 7. VISUALIZACIÓN
# =============================================================================
p = plot(img, axis=false, title="Detección Separada Automatizada")

if !isnothing(bb_simbolo)
    r0, r1, c0, c1 = bb_simbolo
    plot!(p, [c0, c1, c1, c0, c0], [r0, r0, r1, r1, r0],
          color=:blue, linewidth=3, label="Símbolo")
    println("Símbolo → filas $r0:$r1  cols $c0:$c1")
end

if !isnothing(bb_texto)
    r0, r1, c0, c1 = bb_texto
    plot!(p, [c0, c1, c1, c0, c0], [r0, r0, r1, r1, r0],
          color=:red, linewidth=3, label="Texto adidas")
    println("Texto   → filas $r0:$r1  cols $c0:$c1")
    println("Detección: Símbolo + Texto ✓")
else
    println("Detección: Solo Símbolo")
end

savefig(p, "preview_zonas_separadas.png")
println("Imagen guardada como preview_zonas_separadas.png")