import cv2
import numpy as np
import os

# Carregar o classificador Haar Cascade para detecção de rostos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar o reconhecedor facial (LBPH)
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Diretório de treinamento
dataset_dir = "faces_dataset"

# Função para preparar o dataset de treinamento
def preparar_dataset(diretorio):
    faces = []
    ids = []
    for nome_pasta in os.listdir(diretorio):
        caminho_pasta = os.path.join(diretorio, nome_pasta)
        if not os.path.isdir(caminho_pasta):
            continue
        label_id = int(nome_pasta.split('_')[0])
        for imagem_nome in os.listdir(caminho_pasta):
            caminho_imagem = os.path.join(caminho_pasta, imagem_nome)
            imagem = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
            if imagem is None:
                continue
            faces.append(imagem)
            ids.append(label_id)
    return faces, np.array(ids)

# Treinar o reconhecedor com o dataset preparado
faces, ids = preparar_dataset(dataset_dir)
recognizer.train(faces, ids)

# Captura de vídeo em tempo real
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Converter a imagem para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar faces no frame atual
    faces_detectadas = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in faces_detectadas:
        # Extração da região da face
        face_roi = gray[y:y+h, x:x+w]

        # Reconhecimento facial utilizando LBPH
        id_pred, confianca = recognizer.predict(face_roi)

        # Definir nome para cada ID conhecido
        nome = "Desconhecido"
        if confianca < 50:
            if id_pred == 1:
                nome = "Paulo"
            elif id_pred == 2:
                nome = "Maria"

        # Exibir resultados
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'{nome} ({int(confianca)}%)', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Mostrar o frame com reconhecimento
    cv2.imshow('Reconhecimento Facial em Tempo Real', frame)

    # Encerrar com 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
