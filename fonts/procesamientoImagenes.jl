using Images
using FileIO
using Statistics


function loadFolderImages(folderPath::String)
    imagePaths = String[]
    imageLabels = String[]

    for (root, dirs, files) in walkdir(folderPath)
        for file in files
            if endswith(lowercase(file), ".jpg")
                push!(imagePaths, joinpath(root, file))
                push!(imageLabels, basename(root))
            end
        end
    end

    @assert(!isempty(imagePaths), "No se encontraron imágenes en: $folderPath")
    println("Imágenes cargadas: ", length(imagePaths))

    features = [extractImageFeatures(FileIO.load(path)) for path in imagePaths]
    inputs = reduce(vcat, [f' for f in features])

    return (inputs, imageLabels)
end

function extractImageFeatures(img)::Vector{Float64}
    grayImg = Float64.(Gray.(img))
    rows, cols = size(grayImg)

    # 1. UMBRAL DINÁMICO
    umbral = mean(grayImg)
    mascara = grayImg .< (umbral * 0.9) # Buscamos lo que sea un poco más oscuro que la media
    
    # Si la máscara está casi vacía, quizás el logo es blanco sobre fondo oscuro (Invertimos)
    if sum(mascara) < (rows * cols * 0.05)
        mascara = grayImg .> (umbral * 1.1)
    end

    filasConLogo = vec(any(mascara, dims=2))
    colsConLogo  = vec(any(mascara, dims=1))

    # Encontrar la caja delimitadora (Bounding Box)
    filaInicio = something(findfirst(filasConLogo), 1)
    filaFin    = something(findlast(filasConLogo),  rows)
    colInicio  = something(findfirst(colsConLogo),  1)
    colFin     = something(findlast(colsConLogo),   cols)

    logoAlto  = max(1, filaFin - filaInicio)
    logoAncho = max(1, colFin - colInicio)

    caracteristicas = Float64[]

    # 2. CUADRÍCULA 3x3 ADAPTATIVA
    for i in 0:2
        for j in 0:2
            # Calcular límites de cada celda dinámicamente
            r_ini = filaInicio + round(Int, i * logoAlto / 3)
            r_fin = i == 2 ? filaFin : filaInicio + round(Int, (i+1) * logoAlto / 3) - 1
            
            c_ini = colInicio + round(Int, j * logoAncho / 3)
            c_fin = j == 2 ? colFin : colInicio + round(Int, (j+1) * logoAncho / 3) - 1
            
            # Seguros anti-desbordamiento
            r_ini = max(1, min(r_ini, rows)); r_fin = max(r_ini, min(r_fin, rows))
            c_ini = max(1, min(c_ini, cols)); c_fin = max(c_ini, min(c_fin, cols))

            celda = grayImg[r_ini:r_fin, c_ini:c_fin]
            
            # Añadir media y desviación típica de la celda
            push!(caracteristicas, mean(celda))
            s = std(celda)
            push!(caracteristicas, isnan(s) ? 0.0 : s) # Evitar NaN si la celda es de 1 píxel
        end
    end

    # 3. CARACTERÍSTICAS GLOBALES
    push!(caracteristicas, logoAncho / logoAlto) # Relación de aspecto (Aspect Ratio)
    push!(caracteristicas, sum(mascara[filaInicio:filaFin, colInicio:colFin]) / (logoAlto * logoAncho)) # Densidad de píxeles

    return caracteristicas
end