<h1 align="center">Closed-AI<h1/>

<h3 align="center">Closed-AI is a FiveM AI Project<h3/>

<p>

Doc : main

    Importation des bibliothèques nécessaires : Le script commence par importer TensorFlow, Keras et Matplotlib.

    Définition de la couche Involution : Cette couche personnalisée prend plusieurs paramètres lors de son initialisation, tels que le nombre de canaux, le nombre de groupes, la taille du noyau, la taille de la foulée (stride) et le ratio de réduction. La couche construit sa structure dans la méthode build, en créant diverses sous-couches pour le pooling moyen, la génération de noyaux et la mise en forme. La méthode call implémente l'opération principale de l'involution.

    Création d'un tenseur d'entrée : Un tenseur d'entrée aléatoire de forme (32, 256, 256, 3) est généré.

    Calcul de l'involution avec différentes configurations : Trois instances de la couche Involution sont appelées avec différentes configurations pour démontrer son utilisation. Les formes de sortie pour chaque configuration sont affichées.

    Chargement de l'ensemble de données CIFAR-10 : L'ensemble de données CIFAR-10 est chargé, normalisé, mélangé et mis en lots. Les noms de classe sont définis à des fins de visualisation.

    Construction et entraînement du modèle Convolution : Un modèle de réseau de neurones convolutifs (CNN) est construit en utilisant des couches Conv2D et Dense. Il est ensuite compilé et entraîné en utilisant l'ensemble de données CIFAR-10.

    Construction et entraînement du modèle Involution : Un modèle basé sur l'involution est défini. Il contient des couches Involution entremêlées avec des couches ReLU et MaxPooling. Ce modèle est également compilé et entraîné en utilisant l'ensemble de données CIFAR-10.

    Résumés des modèles : Le script affiche des résumés des modèles Convolution et Involution.

    Traçage de l'historique de l'entraînement : Matplotlib est utilisé pour créer des graphiques montrant la perte et la précision au fil des époques d'entraînement pour les deux modèles.

    Visualisation des noyaux d'involution : Le script visualise les noyaux appris par les couches d'involution en réponse à dix images d'exemple provenant de l'ensemble de test.

Veuillez noter que ce script est destiné à être exécuté dans un environnement Python avec TensorFlow et les dépendances requises installées. Il démontre l'utilisation de la couche d'involution dans le contexte de la classification d'images sur l'ensemble de données CIFAR-10 et compare les performances à un modèle convolutif traditionnel.

</p>