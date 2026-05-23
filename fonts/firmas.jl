
# Tened en cuenta que en este archivo todas las funciones tienen puesta la palabra reservada 'function' y 'end' al final
# Según cómo las defináis, podrían tener que llevarlas o no

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 2 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Statistics
using Flux 
using Flux.Losses
using LinearAlgebra


function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}) 
    
    #analizamos la cantidad de clases que hay.
    numClasses = length(classes);
    
    if numClasses<=2 #en el caso de solo 2 clases
         return reshape(feature.==classes[1], :, 1);
    else # Si hay mas de dos clases se genera una matriz con una columna por clase, tenemos tantas posibilidades de hacerlo como nos indican en la solución del ejercicio 1,
        #elegiremos el método sin bucles:
        return convert(BitArray{2}, hcat([instance.==classes for instance in feature]...)');
    end;
    
end;

function oneHotEncoding(feature::AbstractArray{<:Any,1})          

    #analizaremos por nuestra cuenta las clases que hay en el vector de características
    classes= unique(feature);

    #utilizaremos la función anterior para hacer el one hot encoding(sabiendo que la función esta sobre cargada), diferenciando solo en que no le pasamos las clases como parámetro.
    return oneHotEncoding(feature, classes);

end;

function oneHotEncoding(feature::AbstractArray{Bool,1})        

    #en el caso de que el vector de características sea booleano, sabemos que solo hay 2 clases.
    return reshape(feature, :, 1);

end;

function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})      
    
    #devolvemos directamente cada minimo y máximo de cada columna de la matriz de datos.
    return (minimum(dataset, dims=1), maximum(dataset, dims=1));

end;

function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})        

    #devolvemos directamente la media y desviación típica de cada columna de la matriz de datos.
    return (mean(dataset, dims=1), std(dataset, dims=1));

end;

function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})     
    
    # Ya nos dan los valores de normalizacion, por lo que lo que tenemos que hacer es leer esa tupla
    minValues = normalizationParameters[1];
    maxValues = normalizationParameters[2];

    dataset .-= minValues;
    dataset ./= (maxValues .- minValues);

    # Si hay algun atributo en el que todos los valores son iguales, se pone a 0
    dataset[:, vec(minValues.==maxValues)] .= 0;

    return dataset

end;

function normalizeMinMax!(dataset::AbstractArray{<:Real,2})     

    #Llamamos a la funciona para normalizar los valores
    normalizationParameters = calculateMinMaxNormalizationParameters(dataset);
    # A continuacion, lo unico que hay que hacer es llamar a la funcion anterior.
    return normalizeMinMax!(dataset, normalizationParameters);
    
end;

function normalizeMinMax(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})      
    # se copian los datos originales para no modificar el dataset de entrada
    newDataset = copy(dataset)
    
    # llamamos a la funcion de antes que modifica la copia del dataset
    normalizeMinMax!(newDataset, normalizationParameters)
    
    return newDataset
end;

function normalizeMinMax(dataset::AbstractArray{<:Real,2})    
    # calcular parametros
    normalizationParameters = calculateMinMaxNormalizationParameters(dataset)
    
    # llamamos a la funciona anterior
    return normalizeMinMax(dataset, normalizationParameters)
end;

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})     
    avgValues = normalizationParameters[1];
    stdValues = normalizationParameters[2];

    dataset .-= avgValues;
    dataset ./= stdValues;

    dataset[:, vec(stdValues .== 0)] .= 0;

    return dataset;
end;

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})    
    normalizationParameters = calculateZeroMeanNormalizationParameters(dataset);

    return normalizeZeroMean!(dataset, normalizationParameters);
end;

function normalizeZeroMean(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})      
    newDataset = copy(dataset)

    normalizeZeroMean!(newDataset, normalizationParameters)

    return newDataset
end;    

function normalizeZeroMean(dataset::AbstractArray{<:Real,2})    
    normalizationParams = calculateZeroMeanNormalizationParameters(dataset);

    return normalizeZeroMean(dataset, normalizationParams);
end;

function classifyOutputs(outputs::AbstractArray{<:Real,1}; threshold::Real=0.5)     
    
    return outputs .>= threshold
end;

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)     
    
    if size(outputs, 2) == 1 #Si es clasificacion binaria
        return reshape(classifyOutputs(outputs[:]; threshold), :, 1) #Cambiamos de matriz a vec, lo pasamos a la fun anterior y volvemos a cambiarlo a matriz

    else #Si es clasificacion multiclase
        (_, indicesMaxEachInstance) = findmax(outputs, dims=2);  #Buscamos los maximos
        output = falses(size(outputs)); #Cramos una matriz llena de falses
        output[indicesMaxEachInstance] .= true; #Ponemos en true solo dnd encontramos maximos

        return output #lo llamamos output en vez de outputs para no confundirlo el parametro de entrada
    end
end;

function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) 
    return mean(outputs .== targets)    #la precisión es la media del numero de coindidencias entre entradas y salidas
end;

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    if size(outputs,2)==1 
        return accuracy(outputs[:,1], targets[:,1])     # en el caso de que solo haya una coluna se llama a la funciona anterior

    else # se compara fila a fila si son iguales usando eachrow y lo guardamos en un verctor de booleanos 
        class_comparison=eachrow(outputs) .==eachrow(targets); 
        return mean(class_comparison) #devolvemos la media aciertos
    end
end;

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)        
    outputs_bool = classifyOutputs(outputs; threshold);

    return accuracy(outputs_bool, targets);
end;

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)        
    if size(outputs, 2) == 1
        return accuracy(outputs[:, 1], targets[:, 1]; threshold=threshold);
        
    else
        outputs_bool = classifyOutputs(outputs);

        return accuracy(outputs_bool, targets);
    end
end;

function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))     
    
    ann=Chain();
    numInputsLayer=numInputs
    numOutputsLayer=numOutputs
    if !isempty(topology)
        for (i,numOutputsLayer) in enumerate(topology)
            ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, transferFunctions[i]) )
            numInputsLayer = numOutputsLayer
        end
    end

    if numOutputs==1 ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, σ) );
    else ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, identity), softmax) end
    return ann
end;

function trainClassANN(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    inputs =Float32.(dataset[1]);
    targets = (dataset[2]); 

    numInputs = size(inputs, 2);
    numOutputs = size(targets, 2);

    ann = buildClassANN(numInputs, topology, numOutputs; transferFunctions=transferFunctions);

    loss(model,x,y)= (size(y,1)==1) ? Flux.Losses.binarycrossentropy(model(x), y) : Flux.Losses.crossentropy(model(x), y);

    opt_state = Flux.setup(Adam(learningRate), ann);

    inputs_t = inputs';
    targets_t = targets';

    data = [(inputs_t, targets_t)];

    loss_history = Float32[];

    current_loss = loss(ann, inputs_t, targets_t);
    push!(loss_history, current_loss);

    for epoch in 1:maxEpochs
        if current_loss <= minLoss
            break
        end

        Flux.train!(loss, ann, data, opt_state);
        current_loss = loss(ann, inputs_t, targets_t);
        push!(loss_history, current_loss);
    end

    return (ann, loss_history);
end;

function trainClassANN(topology::AbstractArray{<:Int,1}, (inputs, targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    targets_matrix = reshape(targets, :, 1);
    return trainClassANN(topology, (inputs, targets_matrix); 
        transferFunctions=transferFunctions, 
        maxEpochs=maxEpochs, 
        minLoss=minLoss, 
        learningRate=learningRate);
end;


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 3 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random

function holdOut(N::Int, P::Real)
    indices = randperm(N); #índices aleatorios 

    patrones = Int(floor(N * P)); #usamos porcentaje P para saber el nº patrones del test

    # dividimos los índices (los primeros para entrenamiento y los últimos para test)
    trainingIndices = indices[1:end-patrones];
    testIndices = indices[end-patrones+1:end];

    return (trainingIndices, testIndices);
end;

function holdOut(N::Int, Pval::Real, Ptest::Real)
    (indices_resto, indices_test)=holdOut(N, Ptest)

    numPatronesVal = Int(floor(N * Pval)); # 

    Pnueva = numPatronesVal / length(indices_resto);

    (indices_entrenamiento_relativo, indices_resto_relativo)=holdOut(length(indices_resto), Pnueva);

    return(indices_resto[indices_entrenamiento_relativo], indices_resto[indices_resto_relativo], indices_test)
end;

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0,size(trainingDataset[2],2))),
    testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0,size(trainingDataset[2],2))),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)

    # 1. Preparación de datos: Convertir a Float32 y Trasponer
    # Flux espera que los patrones estén en columnas
    inputs_train = Float32.(trainingDataset[1])';
    targets_train = trainingDataset[2]';
    
    inputs_val = Float32.(validationDataset[1])';
    targets_val = validationDataset[2]';
    
    inputs_test = Float32.(testDataset[1])';
    targets_test = testDataset[2]';

    # Detectar si tenemos validación y test (si no están vacíos)
    has_val = size(inputs_val, 2) > 0;
    has_test = size(inputs_test, 2) > 0;

    # 2. Construcción de la RNA
    numInputs = size(inputs_train, 1);
    numOutputs = size(targets_train, 1);
    ann = buildClassANN(numInputs, topology, numOutputs; transferFunctions=transferFunctions);

    # 3. Configuración del entrenamiento
    loss(model,x,y)= (size(y,1)==1) ? Flux.Losses.binarycrossentropy(model(x), y) : Flux.Losses.crossentropy(model(x), y);
    opt_state = Flux.setup(Adam(learningRate), ann);
    
    # Batch único con todos los datos de entrenamiento
    data = [(inputs_train, targets_train)];

    # 4. Inicialización de vectores de histórico
    trainingLosses = Float32[];
    validationLosses = Float32[];
    testLosses = Float32[];

    # 5. Cálculo del Ciclo 0 (Antes de entrenar)
    push!(trainingLosses, loss(ann, inputs_train, targets_train));
    if has_val push!(validationLosses, loss(ann, inputs_val, targets_val)) end;
    if has_test push!(testLosses, loss(ann, inputs_test, targets_test)) end;

    # Variables para Early Stopping
    best_val_loss = has_val ? validationLosses[1] : Inf32;
    best_ann = deepcopy(ann); # Guardamos copia inicial
    epochsWithoutImprovement = 0;

    # 6. Bucle de Entrenamiento
    for epoch in 1:maxEpochs
        # Entrenar 1 ciclo
        Flux.train!(loss, ann, data, opt_state);

        # Calcular losses actuales
        current_train_loss = loss(ann, inputs_train, targets_train);
        push!(trainingLosses, current_train_loss);

        if has_test
            push!(testLosses, loss(ann, inputs_test, targets_test));
        end

        if has_val
            current_val_loss = loss(ann, inputs_val, targets_val);
            push!(validationLosses, current_val_loss);

            # Lógica de Parada Temprana
            if current_val_loss < best_val_loss
                best_val_loss = current_val_loss;
                best_ann = deepcopy(ann);
                epochsWithoutImprovement = 0;
            else
                epochsWithoutImprovement += 1;
            end

            if epochsWithoutImprovement >= maxEpochsVal
                break;
            end
        else
            # Si no hay validación, el criterio de parada es el minLoss de entrenamiento
            # y la mejor red es simplemente la actual.
            best_ann = ann; 
            if current_train_loss <= minLoss
                break;
            end
        end
    end

    # Si no hubo validación, devolvemos la red del último ciclo (ann).
    # Si hubo validación, devolvemos la mejor encontrada (best_ann).
    final_ann = has_val ? best_ann : ann;

    return (final_ann, trainingLosses, validationLosses, testLosses);
end;

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0)),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)
    
    # Convertir vectores de salida a matrices columna (reshape) para los 3 conjuntos
    train_y_matrix = reshape(trainingDataset[2], :, 1);
    val_y_matrix   = reshape(validationDataset[2], :, 1);
    test_y_matrix  = reshape(testDataset[2], :, 1);

    # Reconstruir las tuplas
    trainingDataset_m   = (trainingDataset[1], train_y_matrix);
    validationDataset_m = (validationDataset[1], val_y_matrix);
    testDataset_m       = (testDataset[1], test_y_matrix);

    # Llamar a la función principal
    return trainClassANN(topology, trainingDataset_m;
        validationDataset=validationDataset_m,
        testDataset=testDataset_m,
        transferFunctions=transferFunctions,
        maxEpochs=maxEpochs,
        minLoss=minLoss,
        learningRate=learningRate,
        maxEpochsVal=maxEpochsVal);
end;



# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 4 --------------------------------------------
# ----------------------------------------------------------------------------------------------


function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    VP = sum(outputs .& targets)
    VN = sum(.!outputs .& .!targets)
    FP = sum(outputs .& .!targets)
    FN = sum(.!outputs .& targets)
    
    acc = accuracy(outputs, targets)
    errorRate = 1.0 - acc
    
    recall = (VP + FN == 0) ? 1.0 : VP / (VP + FN)
    specificity = (VN + FP == 0) ? 1.0 : VN / (VN + FP)
    precision = (VP + FP == 0) ? 1.0 : VP / (VP + FP)
    NPV = (VN + FN == 0) ? 1.0 : VN / (VN + FN)
    
    F1 = (precision + recall == 0.0) ? 0.0 : (2 * precision * recall) / (precision + recall)
    
    confMatrix = [VN FP; 
                  FN VP]
                   
    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix)
end

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    
    outputs_bool = classifyOutputs(outputs; threshold = threshold);
    return confusionMatrix(outputs_bool, targets);

end;

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    
    @assert size(outputs, 2) == size(targets, 2)
    @assert size(outputs, 2) != 2
    
    numClasses = size(outputs, 2)
    
    if numClasses == 1
        return confusionMatrix(outputs[:, 1], targets[:, 1])
    end
    
    recall_vec = zeros(Float64, numClasses)
    specificity_vec = zeros(Float64, numClasses)
    precision_vec = zeros(Float64, numClasses)
    NPV_vec = zeros(Float64, numClasses)
    F1_vec = zeros(Float64, numClasses)
    
    for c in 1:numClasses
        _, _, recall_vec[c], specificity_vec[c], precision_vec[c], NPV_vec[c], F1_vec[c], _ = confusionMatrix(outputs[:, c], targets[:, c])
    end
    
    confMatrix = targets' * outputs
    
    if weighted
        pesos = vec(sum(targets, dims=1))
        total_instancias = sum(pesos)
        
        recall = sum(recall_vec .* pesos) / total_instancias
        specificity = sum(specificity_vec .* pesos) / total_instancias
        precision = sum(precision_vec .* pesos) / total_instancias
        NPV = sum(NPV_vec .* pesos) / total_instancias
        F1 = sum(F1_vec .* pesos) / total_instancias
    else
        recall = mean(recall_vec)
        specificity = mean(specificity_vec)
        precision = mean(precision_vec)
        NPV = mean(NPV_vec)
        F1 = mean(F1_vec)
    end
    
    acc = accuracy(outputs, targets)
    errorRate = 1.0 - acc
    
    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix)
end

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)
    outputs_bool = classifyOutputs(outputs; threshold = threshold);
    return confusionMatrix(outputs_bool, targets; weighted = weighted);
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    @assert (all([in(label, classes) for label in vcat(targets, outputs)]));

    outputs_bool = oneHotEncoding(outputs, classes);
    targets_bool = oneHotEncoding(targets, classes);
    
    return confusionMatrix(outputs_bool, targets_bool; weighted = weighted);
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    classes = unique(vcat(targets, outputs));

    return confusionMatrix(outputs, targets, classes; weighted = weighted);
end;

using SymDoME
using GeneticProgramming


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    trainingInputs = Float64.(trainingDataset[1])
    testInputs_float = Float64.(testInputs)
    trainingTargets = trainingDataset[2]
    
    model = dome(trainingInputs, trainingTargets; maximumNodes = maximumNodes)[1]
    
    testOutputs = evaluateTree(model, testInputs_float)
    
    if isa(testOutputs, Real)
        testOutputs = repeat([testOutputs], size(testInputs, 1))
    end
    
    return testOutputs 
end


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)    
    trainingInputs = trainingDataset[1]
    trainingTargets = trainingDataset[2]
    
    num_columnas = size(trainingTargets, 2)
    num_instancias_test = size(testInputs, 1)
    
    if num_columnas == 1
        vector_targets = vec(trainingTargets)
        salida_vector = trainClassDoME((trainingInputs, vector_targets), testInputs, maximumNodes)
        return reshape(salida_vector, num_instancias_test, 1) 
        
    else
        matriz_salida = zeros(Float64, num_instancias_test, num_columnas) 
        
        for i in 1:num_columnas 
            columna_actual = trainingTargets[:, i]
            salida_columna = trainClassDoME((trainingInputs, columna_actual), testInputs, maximumNodes)
            matriz_salida[:, i] = salida_columna 
        end
        
        return matriz_salida 
    end
end

function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    trainingInputs = trainingDataset[1]
    trainingTargets = trainingDataset[2]
    
    classes = unique(trainingTargets)
    testOutputs = Array{eltype(trainingTargets), 1}(undef, size(testInputs, 1))
    
    # Convierte las etiquetas Any a matriz booleana y llama a la versión 2
    targets_codificados = oneHotEncoding(trainingTargets, classes)
    testOutputsDOME = trainClassDoME((trainingInputs, targets_codificados), testInputs, maximumNodes)
    
    testOutputsBool = classifyOutputs(testOutputsDOME; threshold=0) 
    
    if length(classes) <= 2
        vector_bool = vec(testOutputsBool)
        testOutputs[vector_bool] .= classes[1]
        if length(classes) == 2
            testOutputs[.!vector_bool] .= classes[2]
        end
    else
        for numClass in 1:length(classes)
            testOutputs[testOutputsBool[:, numClass]] .= classes[numClass]
        end
    end
    
    return testOutputs
end


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 5 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random
using Random:seed!

function crossvalidation(N::Int64, k::Int64)
    # vector con k elementos ordenados, desde 1 hasta k.
    numRepeticiones = ceil(Int, N/k)
    # vector con repeticiones hasta que la longitud sea mayor o igual a N
    vectorRepeticiones = repeat(1:k, numRepeticiones)
    # tomar los N primeros valores
    vectorNElementos=vectorRepeticiones[1:N]
    # desordenar este vector y devolverlo
    shuffle!(vectorNElementos)
    return vectorNElementos
end;

function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    N = length(targets)

    # vector de índices
    indices = zeros(Int, N)

    # llamadas func anterior (inst positivas y negativas)
    indices[targets] = crossvalidation(sum(targets), k)
    indices[.!targets] = crossvalidation(sum(.!targets), k)

    return indices
end;

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    N = size(targets, 1)

    # vector de índices
    indices = zeros(Int, N)
    numClases = size(targets, 2)

    for c in 1:numClases
        ## contar elementos, llamar a crossvalidation y asignar
        indices[targets[:, c]] = crossvalidation(sum(targets[:, c]), k)
    end

    return indices
end;

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    targetsOhe = oneHotEncoding(targets)
    
    # devuelve el vector de valores enteros con los índices
    return crossvalidation(targetsOhe, k)
end;

function ANNCrossValidation(topology::AbstractArray{<:Int,1},
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
    crossValidationIndices::Array{Int64,1};
    numExecutions::Int=50,
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, validationRatio::Real=0, maxEpochsVal::Int=20)
    
    # separar dataset e identificar las clases
    inputs, targets_cat = dataset
    classes = unique(targets_cat)
    numClasses = length(classes)
    
    # codificar las salidas deseadas mediante oneHotEncoding pasándole el vector classes
    targets_oh = oneHotEncoding(targets_cat, classes)
    
    # calcular el número de folds
    numFolds = maximum(crossValidationIndices)
    
    # crear vectores para almacenar el promedio de cada métrica en cada fold
    acc_folds = Float64[]
    err_folds = Float64[]
    sen_folds = Float64[]
    esp_folds = Float64[]
    vpp_folds = Float64[]
    vpn_folds = Float64[]
    f1_folds  = Float64[]
    
    # matriz de confusión global con valores a 0 (se irá sumando)
    global_cm = zeros(Float64, numClasses, numClasses)
    
    # bucle externo: iterar por cada fold
    for fold in 1:numFolds
        # índices para entrenamiento y test de este fold
        test_idxs = (crossValidationIndices .== fold)
        train_idxs = .!test_idxs
        
        inputs_train = inputs[train_idxs, :]
        targets_train = targets_oh[train_idxs, :]
        
        inputs_test = inputs[test_idxs, :]
        targets_test = targets_oh[test_idxs, :]
        
        # vectores para recopilar las metricas de las 'numExecutions' dentro de este fold
        acc_execs = Float64[]
        err_execs = Float64[]
        sen_execs = Float64[]
        esp_execs = Float64[]
        vpp_execs = Float64[]
        vpn_execs = Float64[]
        f1_execs  = Float64[]
        
        # array 3D para matrices de confusión de cada ejecución (numClasses x numClasses x numExecutions)
        cm_execs = Array{Float64, 3}(undef, numClasses, numClasses, numExecutions)
        
        # bucle interno: Múltiples entrenamientos para paliar la aleatoriedad de la RNA
        for exec in 1:numExecutions
            if validationRatio > 0
                # calcular el ratio ajustado de validación para el conjunto de entrenamiento actual
                N_total = size(inputs, 1)
                N_train_fold = size(inputs_train, 1)
                val_ratio_fold = (N_total * validationRatio) / N_train_fold
                
                # obtener índices de sub-entrenamiento y validación
                train_val_idxs, val_val_idxs = holdOut(N_train_fold, val_ratio_fold)
                
                x_t = inputs_train[train_val_idxs, :]
                y_t = targets_train[train_val_idxs, :]
                x_v = inputs_train[val_val_idxs, :]
                y_v = targets_train[val_val_idxs, :]
                
                # entrenar RNA con conjunto de validación
                ann = trainClassANN(topology, (x_t, y_t), 
                                    validationDataset=(x_v, y_v),
                                    transferFunctions=transferFunctions,
                                    maxEpochs=maxEpochs, minLoss=minLoss, 
                                    learningRate=learningRate, maxEpochsVal=maxEpochsVal)[1]
            else
                # entrenar RNA sin conjunto de validación
                ann = trainClassANN(topology, (inputs_train, targets_train), 
                                    transferFunctions=transferFunctions,
                                    maxEpochs=maxEpochs, minLoss=minLoss, 
                                    learningRate=learningRate)[1]
            end
            
            # obtener las salidas de la RNA con el conjunto de test 
            outputs_test = ann(inputs_test')'
            
            # calcular métricas con confusionMatrix
            acc, err, sen, esp, vpp, vpn, f1, cm = confusionMatrix(outputs_test, targets_test)
            
            # almacenar resultados de la ejecución
            push!(acc_execs, acc)
            push!(err_execs, err)
            push!(sen_execs, sen)
            push!(esp_execs, esp)
            push!(vpp_execs, vpp)
            push!(vpn_execs, vpn)
            push!(f1_execs, f1)
            
            cm_execs[:, :, exec] = cm
        end
        
        # una vez hechas todas las ejecuciones, se hace la media de las métricas para este fold
        push!(acc_folds, mean(acc_execs))
        push!(err_folds, mean(err_execs))
        push!(sen_folds, mean(sen_execs))
        push!(esp_folds, mean(esp_execs))
        push!(vpp_folds, mean(vpp_execs))
        push!(vpn_folds, mean(vpn_execs))
        push!(f1_folds,  mean(f1_execs))
        
        cm_mean_fold = dropdims(mean(cm_execs, dims=3), dims=3)
        global_cm .+= cm_mean_fold
    end
    
    # devolver una tupla con la media y desviación típica de cada métrica + matriz de confusión global
    return (
        (mean(acc_folds), std(acc_folds)),
        (mean(err_folds), std(err_folds)),
        (mean(sen_folds), std(sen_folds)),
        (mean(esp_folds), std(esp_folds)),
        (mean(vpp_folds), std(vpp_folds)),
        (mean(vpn_folds), std(vpn_folds)),
        (mean(f1_folds),  std(f1_folds)),
        global_cm
    )
end


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 6 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using MLJ
using LIBSVM, MLJLIBSVMInterface
using NearestNeighborModels, MLJDecisionTreeInterface

SVMClassifier = MLJ.@load SVC pkg=LIBSVM verbosity=0
kNNClassifier = MLJ.@load KNNClassifier pkg=NearestNeighborModels verbosity=0
DTClassifier  = MLJ.@load DecisionTreeClassifier pkg=DecisionTree verbosity=0


function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}, crossValidationIndices::Array{Int64,1})

    if modelType == :ANN
        topology = modelHyperparameters["topology"]

        return ANNCrossValidation(topology, dataset, crossValidationIndices; 
            numExecutions= get(modelHyperparameters, "numExecutions", 50),
            transferFunctions= get(modelHyperparameters, "transferFunctions", fill(σ, length(topology))),
            maxEpochs= get(modelHyperparameters, "maxEpochs", 1000),
            minLoss= get(modelHyperparameters, "minLoss", 0.0),
            learningRate= get(modelHyperparameters, "learningRate", 0.01),
            validationRatio= get(modelHyperparameters, "validationRatio", 0.0),
            maxEpochsVal= get(modelHyperparameters, "maxEpochsVal", 20) 
        )
    end

    inputs = dataset[1]
    targets = string.(dataset[2])

    classes = unique(targets)
    numClasses = length(classes)
    numFolds = maximum(crossValidationIndices)

    acc_folds, err_folds, sen_folds, esp_folds, vpp_folds, vpn_folds, f1_folds = [Float64[] for _ in 1:7]
    global_cm = zeros(Float64, numClasses, numClasses)

    for fold in 1:numFolds
        test_idxs = (crossValidationIndices .== fold)
        train_idxs = .!test_idxs

        x_train, y_train = inputs[train_idxs, :], targets[train_idxs]
        x_test, y_test = inputs[test_idxs, :], targets[test_idxs]

        local testOutputs
        
        if modelType== :DoME
            testOutputs = trainClassDoME((x_train, y_train), x_test, modelHyperparameters["maximumNodes"])       
                 
        elseif modelType in [:SVC, :DecisionTreeClassifier, :KNeighborsClassifier]
            
            if modelType == :SVC

                kernels = Dict("linear" => LIBSVM.Kernel.Linear, "rbf" => LIBSVM.Kernel.RadialBasis, 
                               "sigmoid" => LIBSVM.Kernel.Sigmoid, "poly" => LIBSVM.Kernel.Polynomial)
                
                model = SVMClassifier(
                    kernel = kernels[modelHyperparameters["kernel"]],
                    cost   = Float64(modelHyperparameters["C"]),
                    gamma  = Float64(get(modelHyperparameters, "gamma", 0.0)),
                    degree = Int32(get(modelHyperparameters, "degree", 3)),
                    coef0  = Float64(get(modelHyperparameters, "coef0", 0.0))
                )
            elseif modelType == :DecisionTreeClassifier
                model = DTClassifier(max_depth = modelHyperparameters["max_depth"], rng = 1)

            elseif modelType == :KNeighborsClassifier
                model = KNNClassifier(K = modelHyperparameters["n_neighbors"])
            end

            mach = machine(model, MLJ.table(x_train), categorical(y_train))
            MLJ.fit!(mach, verbosity=0)
            
            predictions = MLJ.predict(mach, MLJ.table(x_test))
            
            testOutputs = (modelType == :SVC) ? predictions : mode.(predictions)

        end


        acc, err, sen, esp, vpp, vpn, f1, cm = confusionMatrix(testOutputs, y_test, classes)

        push!(acc_folds, acc)
        push!(err_folds, err)
        push!(sen_folds, sen)
        push!(esp_folds, esp)
        push!(vpp_folds, vpp)
        push!(vpn_folds, vpn)
        push!(f1_folds, f1)

        global_cm .+= cm

    end

    return (
        (mean(acc_folds), std(acc_folds)),
        (mean(err_folds), std(err_folds)),
        (mean(sen_folds), std(sen_folds)),
        (mean(esp_folds), std(esp_folds)),
        (mean(vpp_folds), std(vpp_folds)),
        (mean(vpn_folds), std(vpn_folds)),
        (mean(f1_folds),  std(f1_folds)),
        global_cm
    )

end;



