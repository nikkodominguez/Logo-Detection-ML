include("./fonts/firmas.jl")
include("./fonts/procesamientoImagenes.jl")
using Random

# Fijamos la semilla aleatoria 
Random.seed!(123)

# ====================================================================
# 1. Cargar imágenes y normalizar
# ====================================================================
(inputs, targets) = loadFolderImages("./DatasetLogos/train_filtrado_AA")

# Normalizamos los datos extraídos
normalizeMinMax!(inputs) 

println("\n" * "="^50)
println("DATOS CARGADOS")
println("="^50)
println("Imágenes cargadas: ", length(targets))
println("Tamaño inputs: ", size(inputs))   
println("Clases encontradas: ", unique(targets)) 

# ====================================================================
# 2. Crear índices de validación cruzada
# ====================================================================
k = 10
targets_bool = targets .== "adidas_fake"
crossValIdx = crossvalidation(targets_bool, k)

# ====================================================================
# 3. BLOQUE DE MODELOS (Generación de Tablas)
# ====================================================================

# --- 3.1 REDES DE NEURONAS ARTIFICIALES ---
println("\n" * "="^60)
println("TABLA 4.1: REDES DE NEURONAS ARTIFICIALES")
println("="^60)
arquitecturas = [[4], [8], [16], [32], [4, 4], [8, 8], [16, 8], [32, 16]]

for arqui in arquitecturas
    resultados = modelCrossValidation(
        :ANN,
        Dict(
            "topology" => arqui,         
            "learningRate" => 0.01,     
            "maxEpochs" => 1000,        
            "validationRatio" => 0.2,   
            "maxEpochsVal" => 20,       
            "numExecutions" => 5        
        ),
        (convert(Array{Float32,2}, inputs), targets),
        crossValIdx
    )
    
    (acc, errRate, recall, spec, prec, npv, f1, cm) = resultados
    println("Arquitectura $arqui -> Acc: $(round(acc[1], digits=4)) ± $(round(acc[2], digits=4)) | F1: $(round(f1[1], digits=4)) ± $(round(f1[2], digits=4)) | Recall: $(round(recall[1], digits=4)) ± $(round(recall[2], digits=4))")
end


# --- 3.2 MÁQUINAS DE VECTORES SOPORTE - SVM  ---
println("\n" * "="^60)
println("TABLA 4.2: MÁQUINAS DE VECTORES SOPORTE (SVM)")
println("="^60)
configs_svm = [
    ("linear", 0.1), ("linear", 1.0), ("linear", 10.0),
    ("rbf", 0.1), ("rbf", 1.0), ("rbf", 10.0),
    ("poly", 1.0), ("poly", 10.0)
]

for (krnl, c_val) in configs_svm
    resultados = modelCrossValidation(
        :SVC,
        Dict(
            "kernel" => krnl,  
            "C" => c_val             
        ),
        (convert(Array{Float32,2}, inputs), targets),
        crossValIdx
    )
    
    (acc, errRate, recall, spec, prec, npv, f1, cm) = resultados
    println("Kernel $krnl (C=$c_val) -> Acc: $(round(acc[1], digits=4)) ± $(round(acc[2], digits=4)) | F1: $(round(f1[1], digits=4)) ± $(round(f1[2], digits=4)) | Recall: $(round(recall[1], digits=4)) ± $(round(recall[2], digits=4))")
end


# --- 3.3 ÁRBOLES DE DECISIÓN ---
println("\n" * "="^60)
println("TABLA 4.3: ÁRBOLES DE DECISIÓN")
println("="^60)
profundidades = [2, 4, 6, 8, 12, 20]

for prof in profundidades
    resultados = modelCrossValidation(
        :DecisionTreeClassifier,
        Dict(
            "max_depth" => prof  
        ),
        (convert(Array{Float32,2}, inputs), targets),
        crossValIdx
    )
    
    (acc, errRate, recall, spec, prec, npv, f1, cm) = resultados
    println("Profundidad $prof -> Acc: $(round(acc[1], digits=4)) ± $(round(acc[2], digits=4)) | F1: $(round(f1[1], digits=4)) ± $(round(f1[2], digits=4)) | Recall: $(round(recall[1], digits=4)) ± $(round(recall[2], digits=4))")
end


# --- 3.4 K-VECINOS MÁS CERCANOS - kNN ---
println("\n" * "="^60)
println("TABLA 4.4: k-VECINOS MÁS CERCANOS (kNN)")
println("="^60)
vecinos = [1, 3, 5, 7, 11, 15]

for k_vec in vecinos
    resultados = modelCrossValidation(
        :KNeighborsClassifier,
        Dict(
            "n_neighbors" => k_vec 
        ),
        (convert(Array{Float32,2}, inputs), targets),
        crossValIdx
    )
    
    (acc, errRate, recall, spec, prec, npv, f1, cm) = resultados
    println("k=$k_vec -> Acc: $(round(acc[1], digits=4)) ± $(round(acc[2], digits=4)) | F1: $(round(f1[1], digits=4)) ± $(round(f1[2], digits=4)) | Recall: $(round(recall[1], digits=4)) ± $(round(recall[2], digits=4))")
end


# --- 3.5 DoME ---
println("\n" * "="^60)
println("TABLA 4.5: DoME")
println("="^60)
nodos = [10, 20, 30, 40, 50, 60, 80, 100]

for n in nodos
    resultados = modelCrossValidation(
        :DoME,
        Dict(
            "maximumNodes" => n 
        ),
        (convert(Array{Float32,2}, inputs), targets),
        crossValIdx
    )
    
    (acc, errRate, recall, spec, prec, npv, f1, cm) = resultados
    println("Max Nodos $n -> Acc: $(round(acc[1], digits=4)) ± $(round(acc[2], digits=4)) | F1: $(round(f1[1], digits=4)) ± $(round(f1[2], digits=4)) | Recall: $(round(recall[1], digits=4)) ± $(round(recall[2], digits=4))")
end

println("\n" * "="^60)
println("FIN DE LA EJECUCIÓN DE MODELOS")
println("="^60)