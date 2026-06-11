# Résumé des Chapitres 2 et 3
## *Tutorial on Diffusion Models for Imaging and Vision* — Stanley H. Chan (2024)

---

## Introduction

Ce résumé couvre les deux chapitres centraux du tutoriel de Stanley Chan
sur les modèles de diffusion. Le chapitre 2 présente le **Denoising
Diffusion Probabilistic Model (DDPM)**, le cadre fondateur qui transforme
progressivement des images en bruit puis apprend à inverser ce processus.
Le chapitre 3 introduit le **Score-Matching Langevin Dynamics (SMLD)**,
une approche alternative qui apprend le gradient de la log-densité des
données (la *score function*) et l'utilise pour échantillonner via la
dynamique de Langevin. Ensemble, ces deux chapitres constituent les deux
piliers théoriques des modèles de diffusion modernes, unifiés plus tard
dans le chapitre 4 par le formalisme des équations différentielles
stochastiques (SDE).

---

## Chapitre 2 — Denoising Diffusion Probabilistic Model (DDPM)

### 2.1 Blocs de construction

Le DDPM repose sur une chaîne d'états $\mathbf{x}_0, \mathbf{x}_1,
\dots, \mathbf{x}_T$, où $\mathbf{x}_0$ est l'image originale et
$\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$ est du bruit gaussien
pur. Les états intermédiaires $\mathbf{x}_1, \dots, \mathbf{x}_{T-1}$
jouent le rôle de variables latentes.

Le modèle distingue trois types de blocs :

- **Blocs de transition** (états intermédiaires) : chaque état
  $\mathbf{x}_t$ est relié à son prédécesseur par une transition
  *forward* $q_\phi(\mathbf{x}_t | \mathbf{x}_{t-1})$ et à son
  successeur par une transition *reverse* apprise
  $p_\theta(\mathbf{x}_t | \mathbf{x}_{t+1})$.

- **Bloc initial** ($\mathbf{x}_0$) : seule la transition reverse
  $p_\theta(\mathbf{x}_0 | \mathbf{x}_1)$ est définie, puisqu'il n'y a
  pas de prédécesseur.

- **Bloc final** ($\mathbf{x}_T$) : seule la transition forward
  $q_\phi(\mathbf{x}_T | \mathbf{x}_{T-1})$ est définie.

La distribution de transition forward est définie comme :

$$q_\phi(\mathbf{x}_t | \mathbf{x}_{t-1})
  = \mathcal{N}\!\bigl(\mathbf{x}_t \;\big|\;
    \sqrt{\alpha_t}\,\mathbf{x}_{t-1},\;
    (1 - \alpha_t)\,\mathbf{I}\bigr)$$

où $\alpha_t \in (0, 1)$ est un paramètre du *noise schedule*.
Concrètement, cette transition revient à l'opération :

$$\mathbf{x}_t
  = \sqrt{\alpha_t}\,\mathbf{x}_{t-1}
  + \sqrt{1 - \alpha_t}\,\boldsymbol{\epsilon}_t,
  \qquad \boldsymbol{\epsilon}_t \sim \mathcal{N}(0, \mathbf{I})$$

L'image est progressivement atténuée (facteur $\sqrt{\alpha_t} < 1$)
tandis que du bruit indépendant est ajouté. La distribution intermédiaire
évolue d'un mélange gaussien structuré vers une gaussienne isotrope
diffuse.

### 2.2 Les scalaires « magiques » $\sqrt{\alpha_t}$ et $1 - \alpha_t$

Le choix des coefficients $\sqrt{\alpha_t}$ (pour la moyenne) et
$1 - \alpha_t$ (pour la variance) n'est pas arbitraire : il garantit la
**stabilité de la variance** tout au long de la chaîne de diffusion.

Partant d'une variance $\sigma_{t-1}^2$ à l'étape $t-1$, après une
transition la variance devient :

$$\sigma_t^2 = \alpha_t\,\sigma_{t-1}^2 + (1 - \alpha_t)$$

Si $\sigma_0^2 = 1$ et que $\alpha_t$ est constant (par exemple
$\alpha_t = 0{,}97$), la variance converge vers 1 quel que soit le
nombre d'étapes. Plus précisément, la récurrence montre que la variance
est bornée : elle ne diverge pas (pas d'explosion du signal) et ne
s'annule pas (le bruit ne disparaît pas). En concevant un *schedule*
$\{\alpha_t\}_{t=1}^T$ décroissant, on s'assure que le contenu
informationnel de l'image se dégrade *graduellement* et de manière
contrôlée, rendant le processus inverse (du bruit vers l'image) tractable.

### 2.3 Distribution $q_\phi(\mathbf{x}_t | \mathbf{x}_0)$

Plutôt que d'appliquer les transitions une par une, on peut calculer
directement la distribution de $\mathbf{x}_t$ conditionnée à l'image
originale $\mathbf{x}_0$. En développant la récurrence
$\mathbf{x}_t = \sqrt{\alpha_t}\,\mathbf{x}_{t-1}
+ \sqrt{1-\alpha_t}\,\boldsymbol{\epsilon}_t$
jusqu'à $\mathbf{x}_0$, et en utilisant le fait que la somme de
gaussiennes indépendantes reste gaussienne, on obtient :

$$q_\phi(\mathbf{x}_t | \mathbf{x}_0)
  = \mathcal{N}\!\bigl(\mathbf{x}_t \;\big|\;
    \sqrt{\bar\alpha_t}\,\mathbf{x}_0,\;
    (1 - \bar\alpha_t)\,\mathbf{I}\bigr)$$

où $\bar\alpha_t = \prod_{s=1}^{t} \alpha_s$ est le produit cumulé.

Ce résultat est fondamental pour l'entraînement : au lieu de simuler
séquentiellement $T$ étapes, on échantillonne directement
$\mathbf{x}_t$ à partir de $\mathbf{x}_0$ grâce à :

$$\mathbf{x}_t
  = \sqrt{\bar\alpha_t}\,\mathbf{x}_0
  + \sqrt{1 - \bar\alpha_t}\,\boldsymbol{\epsilon},
  \qquad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$

Le coefficient $\sqrt{\bar\alpha_t}$ décroît au fil du temps (le signal
s'atténue), tandis que $1 - \bar\alpha_t$ croît vers 1 (le bruit domine).
À $t = T$, avec un schedule bien choisi, $\bar\alpha_T \approx 0$ et
$\mathbf{x}_T$ est approximativement un bruit gaussien standard.

### 2.4 Evidence Lower Bound (ELBO)

Comme pour le VAE, l'objectif d'entraînement est de maximiser la
vraisemblance des données, bornée inférieurement par l'ELBO :

$$\text{ELBO}
  = \mathbb{E}_{q_\phi(\mathbf{x}_{1:T}|\mathbf{x}_0)}
    \bigl[\log p_\theta(\mathbf{x}_0 | \mathbf{x}_{1:T})\bigr]
  - D_{\text{KL}}\!\bigl(
    q_\phi(\mathbf{x}_{1:T}|\mathbf{x}_0)
    \;\|\;
    p(\mathbf{x}_{1:T})\bigr)$$

Par manipulation algébrique (règle de chaîne et produits télescopiques),
cette expression se décompose en :

$$\text{ELBO}
  = \underbrace{-D_{\text{KL}}\!\bigl(
      q_\phi(\mathbf{x}_T|\mathbf{x}_0) \;\|\; p(\mathbf{x}_T)
    \bigr)}_{\text{terme prior}}
  - \sum_{t=2}^{T}
    \underbrace{D_{\text{KL}}\!\bigl(
      q_\phi(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)
      \;\|\;
      p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)
    \bigr)}_{\text{termes de consistance}}$$

Le **terme prior** compare la marginale à $t = T$ au prior gaussien ; il
est typiquement négligeable si le schedule est bien réglé. Les **termes
de consistance** forment la somme principale : chacun mesure la
divergence KL entre la vraie distribution postérieure
$q_\phi(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)$ (connue
analytiquement, car le processus forward est fixé) et la distribution
reverse apprise $p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)$.

L'observation clé est que le processus forward $q$ ne contient aucun
paramètre apprenable — seul le processus reverse $p_\theta$ est
entraîné. L'ELBO guide l'apprentissage : minimiser la divergence KL
entre la postérieure vraie et les transitions reverse apprises à chaque
pas.

### 2.5 Réécriture du terme de consistance

Pour rendre l'ELBO calculable, il faut une expression explicite de la
postérieure vraie $q_\phi(\mathbf{x}_{t-1} | \mathbf{x}_t,
\mathbf{x}_0)$. En appliquant la règle de Bayes :

$$q_\phi(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)
  = \frac{
    q_\phi(\mathbf{x}_t | \mathbf{x}_{t-1})\;
    q_\phi(\mathbf{x}_{t-1} | \mathbf{x}_0)
  }{
    q_\phi(\mathbf{x}_t | \mathbf{x}_0)
  }$$

Puisque les trois distributions au numérateur et au dénominateur sont
gaussiennes, leur ratio l'est aussi. On obtient :

$$q_\phi(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)
  = \mathcal{N}\!\bigl(\mathbf{x}_{t-1} \;\big|\;
    \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0),\;
    \tilde\beta_t\,\mathbf{I}\bigr)$$

avec la **moyenne postérieure** :

$$\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0)
  = \frac{\sqrt{\alpha_t}\,(1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t}
    \,\mathbf{x}_t
  + \frac{\sqrt{\bar\alpha_{t-1}}\,(1 - \alpha_t)}{1 - \bar\alpha_t}
    \,\mathbf{x}_0$$

et la **variance postérieure** :

$$\tilde\beta_t
  = \frac{(1 - \alpha_t)(1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t}$$

La moyenne postérieure est une combinaison pondérée de deux termes : une
prédiction basée sur l'observation bruitée courante $\mathbf{x}_t$, et
une information sur l'image originale $\mathbf{x}_0$. Les poids
dépendent des paramètres du schedule. La variance, elle, est
*indépendante des données* — elle ne dépend que du schedule.

### 2.6 Dérivation de $q_\phi(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)$

Cette sous-section fournit la dérivation complète de la postérieure par
complétion du carré. On part de la règle de Bayes :

$$q_\phi(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)
  \propto q_\phi(\mathbf{x}_t | \mathbf{x}_{t-1})\;
          q_\phi(\mathbf{x}_{t-1} | \mathbf{x}_0)$$

En substituant les distributions connues :

$$q_\phi(\mathbf{x}_t | \mathbf{x}_{t-1})
  = \mathcal{N}(\mathbf{x}_t | \sqrt{\alpha_t}\,\mathbf{x}_{t-1},
    (1-\alpha_t)\mathbf{I})$$

$$q_\phi(\mathbf{x}_{t-1} | \mathbf{x}_0)
  = \mathcal{N}(\mathbf{x}_{t-1} | \sqrt{\bar\alpha_{t-1}}\,\mathbf{x}_0,
    (1-\bar\alpha_{t-1})\mathbf{I})$$

Le produit de deux gaussiennes est proportionnel à une gaussienne. En
développant l'exposant :

$$\exp\!\Biggl(
  -\frac{(\mathbf{x}_t - \sqrt{\alpha_t}\,\mathbf{x}_{t-1})^2}
        {2(1-\alpha_t)}
  -\frac{(\mathbf{x}_{t-1} - \sqrt{\bar\alpha_{t-1}}\,\mathbf{x}_0)^2}
        {2(1-\bar\alpha_{t-1})}
\Biggr)$$

et en réarrangeant comme un polynôme quadratique en $\mathbf{x}_{t-1}$,
la complétion du carré révèle la moyenne et la variance de la
postérieure donnée en section 2.5. Cette dérivation confirme que la
variance postérieure $\tilde\beta_t$ s'obtient comme l'inverse de la
somme des précisions (inverses des variances) des deux distributions :

$$\tilde\beta_t
  = \frac{1}{\frac{1}{1-\alpha_t} + \frac{1}{1-\bar\alpha_{t-1}}}
  = \frac{(1-\alpha_t)(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}$$

### 2.7 Entraînement et inférence

**Entraînement.** Le processus reverse est paramétré comme :

$$p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)
  = \mathcal{N}\!\bigl(\mathbf{x}_{t-1} \;\big|\;
    \boldsymbol{\mu}_\theta(\mathbf{x}_t, t),\;
    \sigma_t^2\,\mathbf{I}\bigr)$$

La variance $\sigma_t^2$ peut être fixée (souvent $\tilde\beta_t$ ou
$1 - \alpha_t$) ; seule la moyenne
$\boldsymbol{\mu}_\theta(\mathbf{x}_t, t)$ est prédite par un réseau
de neurones.

Plutôt que de prédire directement la moyenne, on reparamétrise le
réseau pour qu'il prédise le **vecteur de bruit**
$\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ :

$$\boldsymbol{\mu}_\theta(\mathbf{x}_t, t)
  = \frac{1}{\sqrt{\alpha_t}}
    \Bigl(\mathbf{x}_t
    - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}
      \,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\Bigr)$$

La **fonction de perte** se simplifie alors en :

$$\boxed{
  \mathcal{L}
  = \mathbb{E}_{t,\,\mathbf{x}_0,\,\boldsymbol{\epsilon}}\!\Bigl[
    \bigl\|\boldsymbol{\epsilon}
    - \boldsymbol{\epsilon}_\theta\!\bigl(
      \sqrt{\bar\alpha_t}\,\mathbf{x}_0
      + \sqrt{1-\bar\alpha_t}\,\boldsymbol{\epsilon},\; t
    \bigr)\bigr\|^2
  \Bigr]
}$$

C'est remarquablement simple : on entraîne un réseau à prédire le bruit
gaussien qui a été ajouté à l'étape $t$. À chaque itération
d'entraînement :

1. Tirer un pas de temps $t$ uniformément dans $\{1, \dots, T\}$.
2. Tirer une image $\mathbf{x}_0$ du dataset.
3. Tirer un bruit $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$.
4. Calculer $\mathbf{x}_t = \sqrt{\bar\alpha_t}\,\mathbf{x}_0
   + \sqrt{1-\bar\alpha_t}\,\boldsymbol{\epsilon}$.
5. Minimiser l'erreur quadratique entre $\boldsymbol{\epsilon}$ et
   $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$.

**Inférence.** On part de $\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$
et on applique itérativement le processus reverse. À chaque pas
$t = T, T-1, \dots, 1$ :

$$\mathbf{x}_{t-1}
  = \frac{1}{\sqrt{\alpha_t}}
    \Bigl(\mathbf{x}_t
    - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}
      \,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\Bigr)
  + \sqrt{\tilde\beta_t}\,\mathbf{z}_t$$

où $\mathbf{z}_t \sim \mathcal{N}(0, \mathbf{I})$ pour $t > 1$ et
$\mathbf{z}_1 = 0$. Après $T$ pas, $\mathbf{x}_0$ est l'image générée.

### 2.8 Dérivation basée sur le vecteur de bruit

Cette section développe la reformulation qui fait de la prédiction du
bruit la cible d'entraînement plutôt que la prédiction de la moyenne.

Partant de $\mathbf{x}_t = \sqrt{\bar\alpha_t}\,\mathbf{x}_0
+ \sqrt{1-\bar\alpha_t}\,\boldsymbol{\epsilon}$, on peut exprimer :

$$\mathbf{x}_0
  = \frac{1}{\sqrt{\bar\alpha_t}}
    \bigl(\mathbf{x}_t - \sqrt{1-\bar\alpha_t}\,\boldsymbol{\epsilon}\bigr)$$

En substituant dans la formule de la moyenne postérieure, on retrouve la
reparamétrisation en termes de bruit. L'équivalence entre la prédiction
de la moyenne et la prédiction du bruit se traduit par :

$$\bigl\|\tilde{\boldsymbol{\mu}}_t
  - \boldsymbol{\mu}_\theta\bigr\|^2
  \propto
  \bigl\|\boldsymbol{\epsilon}
  - \boldsymbol{\epsilon}_\theta\bigr\|^2$$

Ce changement de variable est numériquement plus stable et plus
interprétable : le réseau apprend à estimer le bruit ajouté, ce qui est
souvent plus facile que de prédire des images directement, car les
patterns de bruit sont plus consistants à travers différentes échelles et
types d'images. Cette reparamétrisation explique le succès empirique du
DDPM.

### 2.9 Inversion par débruitage direct (InDI)

La dernière sous-section introduit une technique pour inverser des images
à travers le modèle de diffusion, avec des applications en édition
d'images.

Le principe : étant donnée une image $\mathbf{x}_0$, on l'encode vers
une représentation bruitée $\mathbf{x}_t$ via le processus forward, puis
on décode via le processus reverse appris. Ce processus comprend :

1. **Encodage** : appliquer la diffusion forward pour obtenir
   $\mathbf{x}_t = \sqrt{\bar\alpha_t}\,\mathbf{x}_0
   + \sqrt{1-\bar\alpha_t}\,\boldsymbol{\epsilon}$ pour un $t$ choisi.

2. **Décodage** : appliquer le processus reverse appris itérativement de
   l'étape $t$ à 0.

3. **Édition** : entre l'encodage et le décodage, modifier
   $\mathbf{x}_t$ (par exemple par interpolation vers une autre image au
   même niveau de bruit).

Cette approche fonctionne parce que le modèle de diffusion apprend des
représentations à tous les niveaux de bruit. À des niveaux de bruit
élevés, la structure sémantique est préservée tandis que les détails au
niveau des pixels sont perdus, permettant des opérations significatives
dans cet espace de représentation intermédiaire.

Les applications d'InDI incluent l'inpainting (reconstruction de régions
masquées), le transfert de style, et l'interpolation d'images
(transition fluide entre images par interpolation à des niveaux de bruit
intermédiaires).

---

## Chapitre 3 — Score-Matching Langevin Dynamics (SMLD)

Le chapitre 3 présente une approche fondamentalement différente de la
génération d'images, qui n'utilise ni ELBO ni processus de diffusion
explicite. Au lieu de cela, SMLD repose sur deux idées :

1. **Apprendre le gradient** de la log-densité des données (la *score
   function*) plutôt que la densité elle-même.
2. **Échantillonner** en suivant ce gradient via la dynamique de
   Langevin.

### 3.1 Dynamique de Langevin

La dynamique de Langevin est une technique classique pour échantillonner
à partir de distributions de probabilité complexes. Plutôt que de
construire un modèle génératif explicite, elle utilise des mises à jour
stochastiques itératives guidées par l'information de gradient.

La règle de mise à jour fondamentale est :

$$\mathbf{x}_{t+1}
  = \mathbf{x}_t
  + \frac{\epsilon}{2}\,\nabla_{\mathbf{x}} \log p(\mathbf{x}_t)
  + \sqrt{\epsilon}\,\mathbf{w}_t$$

où :
- $\epsilon$ est un pas (*step size*) petit,
- $\nabla_{\mathbf{x}} \log p(\mathbf{x})$ est la **score function**
  (gradient de la log-probabilité),
- $\mathbf{w}_t \sim \mathcal{N}(0, \mathbf{I})$ est un bruit gaussien
  standard.

L'équation combine trois composantes :

1. Un **terme de dérive déterministe** proportionnel à la score function,
   qui guide le mouvement vers les régions de haute probabilité.
2. Un **terme d'injection de bruit** qui empêche la convergence
   prématurée vers des modes locaux.
3. Des **facteurs d'échelle** soigneusement choisis ($\epsilon/2$ pour la
   dérive, $\sqrt{\epsilon}$ pour le bruit) qui garantissent les
   propriétés de convergence.

**Intuition.** On peut conceptualiser ce processus comme une particule
escaladant une surface de probabilité. La score function agit comme une
boussole pointant vers les sommets. Sans bruit ($\epsilon = 0$), les
particules convergeraient déterministiquement vers les modes. Le bruit
assure l'exploration du paysage entier. Au fil des itérations, avec des
pas appropriés, la distribution empirique des particules converge vers la
distribution cible $p(\mathbf{x})$.

Le défi pratique : l'accès à $\nabla_{\mathbf{x}} \log p(\mathbf{x})$
est typiquement intractable pour les distributions complexes. C'est ce
qui motive les techniques de score matching.

### 3.2 La score function de Stein

La score function, formellement appelée *score de Stein*, est le gradient
de la log-densité de probabilité par rapport à la variable de données :

$$\mathbf{s}(\mathbf{x})
  = \nabla_{\mathbf{x}} \log p(\mathbf{x})$$

**Propriétés fondamentales.** Pour une densité normalisée, l'espérance de
la score function est nulle :

$$\mathbb{E}_{p(\mathbf{x})}[\mathbf{s}(\mathbf{x})]
  = \mathbb{E}_{p(\mathbf{x})}[\nabla_{\mathbf{x}} \log p(\mathbf{x})]
  = \int \nabla_{\mathbf{x}} p(\mathbf{x})\,d\mathbf{x}
  = 0$$

Cette propriété émerge de la commutation de l'intégration et de la
dérivation sous des conditions de régularité appropriées.

**Avantage crucial pour les densités non normalisées.** Si
$p(\mathbf{x}) = \tilde{p}(\mathbf{x}) / Z$ où $Z$ est une constante
de partition intractable, alors :

$$\mathbf{s}(\mathbf{x})
  = \nabla_{\mathbf{x}} \log \tilde{p}(\mathbf{x})
  - \nabla_{\mathbf{x}} \log Z
  = \nabla_{\mathbf{x}} \log \tilde{p}(\mathbf{x})$$

La constante de normalisation disparaît complètement ! Cette propriété
remarquable signifie qu'on peut calculer les scores à partir de densités
non normalisées, évitant le calcul explicite des fonctions de partition —
un avantage pratique considérable.

**Lien avec l'information de Fisher.** La score function est intimement
liée à la matrice d'information de Fisher :

$$\mathbf{F}
  = \mathbb{E}_{p(\mathbf{x})}
    [\mathbf{s}(\mathbf{x})\,\mathbf{s}(\mathbf{x})^T]$$

Cette matrice caractérise la courbure du paysage de probabilité et
apparaît naturellement en théorie de l'apprentissage statistique.

**Interprétation géométrique.** Si l'on imagine la densité de
probabilité comme une carte topographique, la score function est une
boussole en chaque point indiquant la direction de montée la plus raide.
Les régions à gradient de probabilité abrupt produisent des scores
élevés ; les plateaux produisent des scores proches de zéro.

### 3.3 Techniques de score matching

Le score matching résout le problème central : apprendre la score
function à partir de données, sans accès aux vraies densités de
probabilité. Au lieu d'estimer $p(\mathbf{x})$ directement, on entraîne
un réseau de neurones qui approxime $\mathbf{s}(\mathbf{x})$.

**Objectif de score matching.** La fonction de perte fondamentale
minimise :

$$\mathcal{L}_{\text{SM}}
  = \mathbb{E}_{p(\mathbf{x})}\!\bigl[
    \bigl\|\nabla_{\mathbf{x}} \log p(\mathbf{x})
    - \mathbf{s}_\theta(\mathbf{x})\bigr\|^2
  \bigr]$$

où $\mathbf{s}_\theta(\mathbf{x})$ est un réseau paramétré par $\theta$.

**Denoising score matching (DSM).** Le score matching direct est
impraticable car le calcul des vrais scores à partir des données pose des
difficultés fondamentales. Le *denoising score matching* contourne le
problème en introduisant une corruption artificielle par bruit :

$$\mathbf{x}' = \mathbf{x} + \boldsymbol{\epsilon},
  \qquad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \sigma^2\,\mathbf{I})$$

L'objectif de DSM devient :

$$\mathcal{L}_{\text{DSM}}
  = \mathbb{E}_{p(\mathbf{x})}
    \mathbb{E}_{\mathcal{N}(\boldsymbol{\epsilon} | 0, \sigma^2\mathbf{I})}
    \!\bigl[
      \bigl\|\nabla_{\mathbf{x}'} \log p(\mathbf{x}'|\mathbf{x})
      - \mathbf{s}_\theta(\mathbf{x}')\bigr\|^2
    \bigr]$$

Cette formulation est tractable car la distribution conditionnelle
$p(\mathbf{x}'|\mathbf{x}) = \mathcal{N}(\mathbf{x}'|\mathbf{x},
\sigma^2\mathbf{I})$ a une forme analytique explicite. Le score
correspondant est simplement :

$$\nabla_{\mathbf{x}'} \log p(\mathbf{x}'|\mathbf{x})
  = -\frac{\mathbf{x}' - \mathbf{x}}{\sigma^2}$$

On peut démontrer l'**équivalence** entre le denoising score matching et
le score matching explicite sous certaines conditions théoriques. Cette
équivalence justifie rigoureusement l'approche par débruitage.

**Connexion aux autoencodeurs débruiteurs.** Le DSM présente des liens
conceptuels profonds avec les *denoising autoencoders*. Les deux
apprennent à inverser la corruption par bruit, mais le score matching
opère dans l'espace des gradients plutôt que de reconstruire directement
les données propres.

**Score matching multi-échelle.** En pratique, appliquer le score
matching à un seul niveau de bruit est insuffisant. Les implémentations
modernes utilisent **plusieurs échelles de bruit**, entraînant des
réseaux de score $\mathbf{s}_\theta(\mathbf{x}, \sigma)$ qui acceptent
à la fois les données et le niveau de bruit en entrée. Cela génère une
famille de score functions à travers les échelles de bruit, permettant un
échantillonnage robuste à des intensités de bruit variées.

C'est le principe des **Noise-Conditioned Score Networks (NCSN)** : un
unique réseau $\mathbf{s}_\theta(\mathbf{x}, \sigma)$ est entraîné sur
un ensemble de niveaux de bruit $\{\sigma_i\}_{i=1}^L$ avec
$\sigma_1 > \sigma_2 > \dots > \sigma_L$. Le réseau apprend à estimer
le score à chaque échelle simultanément.

**Procédure d'entraînement.** La boucle d'entraînement procède comme
suit :

1. Tirer une donnée réelle $\mathbf{x}$ du dataset.
2. Tirer un niveau de bruit $\sigma$ parmi une distribution prédéfinie
   sur les échelles.
3. Corrompre les données en ajoutant du bruit gaussien :
   $\mathbf{x}' = \mathbf{x}
   + \boldsymbol{\epsilon}$, $\boldsymbol{\epsilon} \sim
   \mathcal{N}(0, \sigma^2\mathbf{I})$.
4. Calculer la perte de denoising score matching.
5. Rétropropager et mettre à jour les paramètres du réseau.

**Inférence par dynamique de Langevin recuite (*annealed Langevin
dynamics*).** Pour l'échantillonnage, on combine les score functions
apprises à différentes échelles avec la dynamique de Langevin. On
commence par un bruit élevé ($\sigma_1$, large) où le paysage de score
est lisse et facile à naviguer, puis on réduit progressivement le niveau
de bruit ($\sigma_1 \to \sigma_2 \to \dots \to \sigma_L$), affinant
l'échantillon à chaque étape. À chaque niveau $\sigma_i$, on exécute
plusieurs pas de Langevin :

$$\mathbf{x}_{t+1}
  = \mathbf{x}_t
  + \frac{\epsilon_i}{2}\,\mathbf{s}_\theta(\mathbf{x}_t, \sigma_i)
  + \sqrt{\epsilon_i}\,\mathbf{w}_t$$

Cette stratégie de recuit résout le problème des modes isolés : à bruit
élevé, les modes sont « connectés » et les particules peuvent naviguer
entre eux ; à bruit faible, chaque mode est finement résolu.

**Importance pour les modèles de diffusion.** Les score functions
apprises permettent à la dynamique de Langevin d'échantillonner
efficacement à partir de distributions complexes. Plutôt que de
spécifier manuellement le modèle de probabilité, on apprend les score
functions purement à partir d'observations. Combiné avec les mises à
jour de Langevin, cela crée des modèles génératifs puissants ne
nécessitant ni modèles de densité explicites ni calculs de vraisemblance
— un changement de paradigme à la base de l'IA générative moderne.

---

## Synthèse et connexions entre les deux chapitres

### Deux perspectives, un même objectif

DDPM (chapitre 2) et SMLD (chapitre 3) attaquent le même problème — la
génération d'échantillons à partir d'une distribution de données complexe
— mais par des voies très différentes :

| Aspect | DDPM | SMLD |
|--------|------|------|
| **Cadre théorique** | ELBO / vraisemblance variationnelle | Score matching |
| **Ce que le réseau prédit** | Bruit $\boldsymbol{\epsilon}$ | Score $\nabla_\mathbf{x} \log p$ |
| **Processus forward** | Chaîne de Markov discrète à $T$ pas | Perturbation par bruit à $L$ échelles |
| **Processus de génération** | Itération reverse : $\mathbf{x}_T \to \mathbf{x}_0$ | Langevin recuit : $\sigma_1 \to \sigma_L$ |
| **Objectif d'entraînement** | MSE sur le bruit | MSE sur le score |
| **Variance** | Fixée par le schedule | Définie par les échelles $\sigma_i$ |

### Équivalence profonde

Malgré ces différences superficielles, les deux approches sont
profondément liées. La connexion clé est que prédire le bruit
$\boldsymbol{\epsilon}$ (DDPM) et estimer le score (SMLD) sont
mathématiquement équivalents à un facteur d'échelle près :

$$\mathbf{s}_\theta(\mathbf{x}_t, t)
  = -\frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}
          {\sqrt{1 - \bar\alpha_t}}$$

Cette équivalence est formalisée dans le chapitre 4 (SDE), qui montre
que DDPM et SMLD sont deux discrétisations d'une même équation
différentielle stochastique, l'une avec un *Variance Preserving (VP)*
SDE, l'autre avec un *Variance Exploding (VE)* SDE.

### Complémentarité

- Le DDPM fournit un cadre d'entraînement simple et stable (prédire le
  bruit avec une MSE) avec une théorie probabiliste rigoureuse (ELBO).
- Le SMLD apporte l'intuition géométrique de la score function et la
  flexibilité de la dynamique de Langevin, notamment la stratégie de
  recuit multi-échelle.
- Les architectures modernes (Stable Diffusion, DALL-E, etc.) combinent
  les deux perspectives : elles utilisent typiquement la paramétrisation
  en bruit du DDPM avec des schedules inspirés du SMLD, le tout dans le
  cadre unifié des SDE.

---

## Conclusion

Les chapitres 2 et 3 du tutoriel de Chan posent les fondations
mathématiques complètes des modèles de diffusion. Le chapitre 2
développe méthodiquement le DDPM, de la définition de la chaîne de
diffusion jusqu'à l'algorithme d'entraînement remarquablement simple
(prédire le bruit ajouté), en passant par les dérivations rigoureuses de
la distribution shortcut, de l'ELBO, et de la postérieure tractable. Le
chapitre 3 introduit l'approche complémentaire par score matching, où
l'on apprend directement le gradient de la log-densité des données et on
l'utilise pour échantillonner via la dynamique de Langevin. L'équivalence
profonde entre ces deux approches — prédire le bruit revient à estimer le
score — constitue l'un des résultats les plus élégants de la théorie des
modèles de diffusion, unifié formellement par le formalisme SDE du
chapitre 4.
