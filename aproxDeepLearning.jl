using Flux
using Flux.Losses
using Flux: onehotbatch, onecold, adjust!
using Images, FileIO
using Statistics: mean, std
using Random

# Cargamos funciones de firmas y procesamiento
include("./fonts/firmas.jl")
include("./fonts/procesamientoImagenes.jl")

# --- CONFIGURACIÓN ---
Random.seed!(1) 
ruta_dataset = "./DatasetLogos/train_CNN"
tamaño_imagen = (64, 64) # Redimensionamos para agilizar las 8 arquitecturas

# =============================================================================
# 1. CARGA Y PREPROCESADO PARA DEEP LEARNING
# =============================================================================

function cargarDatasetDL(path, tamaño)
    imagenes_matrices = []
    etiquetas = String[]
    
    for clase in ["adidas_fake", "adidas_real"]
        folder = joinpath(path, clase)
        for file in readdir(folder)
            if endswith(lowercase(file), ".jpg")
                img = FileIO.load(joinpath(folder, file))
                img_gray = Gray.(img)
                img_res = imresize(img_gray, tamaño)
                
                push!(imagenes_matrices, Float32.(img_res))
                push!(etiquetas, clase)
            end
        end
    end
    
    # Convertir al formato WHCN de Julia: (Ancho, Alto, 1 Canal, N Patrones)
    numPatrones = length(imagenes_matrices)
    inputs = Array{Float32,4}(undef, tamaño[1], tamaño[2], 1, numPatrones)
    for i in 1:numPatrones
        inputs[:,:,1,i] .= imagenes_matrices[i][:,:]'
    end
    
    return inputs, etiquetas
end

# Carga de datos
inputs, labels_str = cargarDatasetDL(ruta_dataset, tamaño_imagen)
# Convertir etiquetas a matriz booleana (Fake = true para priorizar Recall)
targets = reshape(labels_str .== "adidas_fake", 1, :) 

# =============================================================================
# 2. DEFINICIÓN DE LAS 8 ARQUITECTURAS REQUERIDAS
# =============================================================================

arquitecturas = [
    # Config 1: Básica 16-32 filtros
    () -> Chain(Conv((3,3), 1=>16, pad=1, relu), MaxPool((2,2)), Conv((3,3), 16=>32, pad=1, relu), MaxPool((2,2)), x->reshape(x,:,size(x,4)), Dense(8192, 1, σ)),
    # Config 2: Más filtros (32-64)
    () -> Chain(Conv((3,3), 1=>32, pad=1, relu), MaxPool((2,2)), Conv((3,3), 32=>64, pad=1, relu), MaxPool((2,2)), x->reshape(x,:,size(x,4)), Dense(16384, 1, σ)),
    # Config 3: Tres capas convolucionales
    () -> Chain(Conv((3,3), 1=>16, pad=1, relu), MaxPool((2,2)), Conv((3,3), 16=>32, pad=1, relu), MaxPool((2,2)), Conv((3,3), 32=>32, pad=1, relu), MaxPool((2,2)), x->reshape(x,:,size(x,4)), Dense(2048, 1, σ)),
    # Config 4: Filtros más grandes (5x5)[cite: 12]
    () -> Chain(Conv((5,5), 1=>16, pad=2, relu), MaxPool((2,2)), Conv((5,5), 16=>32, pad=2, relu), MaxPool((2,2)), x->reshape(x,:,size(x,4)), Dense(8192, 1, σ)),
    # Config 5: Capa densa intermedia
    () -> Chain(Conv((3,3), 1=>16, pad=1, relu), MaxPool((2,2)), x->reshape(x,:,size(x,4)), Dense(16384, 16, relu), Dense(16, 1, σ)),
    # Config 6: Mayor profundidad (16-32-64)
    () -> Chain(Conv((3,3), 1=>16, pad=1, relu), MaxPool((2,2)), Conv((3,3), 16=>32, pad=1, relu), MaxPool((2,2)), Conv((3,3), 32=>64, pad=1, relu), MaxPool((2,2)), x->reshape(x,:,size(x,4)), Dense(4096, 1, σ)),
    # Config 7: Filtros 3x3 sin pad
    () -> Chain(Conv((3,3), 1=>8, relu), MaxPool((2,2)), Conv((3,3), 8=>16, relu), MaxPool((2,2)), x->reshape(x,:,size(x,4)), Dense(3136, 1, σ)),
    # Config 8: Mínima expresión
    () -> Chain(Conv((3,3), 1=>32, pad=1, relu), MaxPool((4,4)), x->reshape(x,:,size(x,4)), Dense(8192, 1, σ))
]

# =============================================================================
# 3. EXPERIMENTACIÓN CON VALIDACIÓN CRUZADA (10 FOLDS)
# =============================================================================

k_folds = 10
indices = crossvalidation(vec(targets), k_folds)
epocas_por_fold = 20 # Se modifica en función de si tarda mucho o si aprende poco

println("--- Iniciando Experimentación Deep Learning ---")
println("Total de imágenes: ", size(inputs, 4))

for (i, constructora) in enumerate(arquitecturas)
    println("\n=======================================================")
    println("Probando Arquitectura $i...")
    
    # Vectores para guardar los resultados de cada fold
    acc_folds = Float64[]
    f1_folds = Float64[]
    rec_folds = Float64[]
    cm_global = zeros(Float64, 2, 2)
    
    for fold in 1:k_folds
        # 3.1 Separar datos de entrenamiento y test para este fold
        test_idxs = (indices .== fold)
        train_idxs = .!test_idxs
        
        x_train = inputs[:, :, :, train_idxs]
        y_train = targets[:, train_idxs]
        
        x_test = inputs[:, :, :, test_idxs]
        y_test = targets[:, test_idxs]
        
        # 3.2 Construir red y configurar entrenamiento
        ann = constructora()
        opt_state = Flux.setup(Adam(0.01), ann)
        loss(model, x, y) = Losses.binarycrossentropy(model(x), y)
        
        # Un único batch con todo el conjunto de entrenamiento como nos dice el enunciado
        data = [(x_train, y_train)]
        
        # 3.3 Entrenar el modelo
        for epoch in 1:epocas_por_fold
            Flux.train!(loss, ann, data, opt_state)
        end
        
        # 3.4 Evaluar el modelo en el conjunto de test
        outputs_test = ann(x_test)
        
        # Transponemos para que las instancias queden en filas
        acc, err, rec, esp, prec, npv, f1, cm = confusionMatrix(outputs_test', y_test')
        
        push!(acc_folds, acc)
        push!(f1_folds, f1)
        push!(rec_folds, rec)
        cm_global .+= cm
    end
    
    # 3.5 Mostrar resultados finales
    println("\n--- Resultados Arquitectura $i ---")
    println("Accuracy:    ", round(mean(acc_folds), digits=4), " ± ", round(std(acc_folds), digits=4))
    println("F1:          ", round(mean(f1_folds),  digits=4), " ± ", round(std(f1_folds),  digits=4))
    println("Recall:      ", round(mean(rec_folds), digits=4), " ± ", round(std(rec_folds), digits=4))
    println("Matriz de confusión Global (Suma de los 10 folds):")
    println(round.(Int, cm_global))

    GC.gc() # Llama al recolector de basura para vaciar la RAM antes de la siguiente arquitectura
end