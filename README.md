# IA_FramesInterpolation

## Présentation

Dans ce répertoire, vous trouverez un code réalisé en binôme durant ma première année du master ISICG. L'objectif était ici de réaliser une implémentation de la méthode AdaConv : [Video Frame Interpolation via Adaptive Convolution](https://arxiv.org/pdf/1703.07514.pdf). L'idée de la méthode est de passer en entrée de l'IA une vidéo quelconque et d'obtenir en sortie une vidéo avec un frameRate plus élevé. De ce fait, comme la vidéo de sortie contiendra plus d'images que la vidéo d'entrée, l'IA devra générer de toutes nouvelles images qui pourront s'intercaler parfaitement entre les frames de la vidéo.

Si cette méthode vous intrigue je vous conseille d'aller lire le papier SepConv : [Video Frame Interpolation via Adaptive Separable Convolution](https://arxiv.org/pdf/1708.01692.pdf) qui est sorti juste après AdaConv, traite du même sujet et est écrit par les mêmes auteurs.

## Résultats

Dans la vidéo qui suit, une frame sur deux a été générée par notre IA.

![interpolation](results/interpolation.gif)
