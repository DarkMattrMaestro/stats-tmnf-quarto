<a name="readme" href="https://darkmattrmaestro.github.io/stats-tmnf-quarto/Classification_de_toute_allure.pdf"><img src="https://github.com/user-attachments/assets/9a8b8bbf-81d8-4656-b379-b1296cd21b06" alt="Open_Graph_Classification_de_toute_allure"></img></a>

[Click here to see the paper](https://darkmattrmaestro.github.io/stats-tmnf-quarto/Classification_de_toute_allure.pdf). Note that it is only available in french.

Sections 4.2 and 4.3 are the most pertinent.

## General Information

I wanted a better understanding of tags of tracks submitted on [Trackmania Nations Forever Exchange](https://tmnf.exchange/). Thus, for my cumulative project in my statistics course, I ecplored the correlations between a car's path and the track's chosen tag. I established a predictive model that predicts tags, followed by analysing the model to pull conclusions.

The predictive model has a recall of 84.71%, which is rather good considering that no track data was provided to the model. The model has no access to the blocks that compose each track.

## Poster

I created a poster for the <a href="https://islp.ssc.ca">ISLP National Statistics Poster Competition</a> which summarized my project. The poster won second place.

<details>
<summary>See the poster</summary>
<img src="https://github.com/user-attachments/assets/48bf7334-d0a3-4e0a-b563-7730d1cfb67b" alt="Poster"></img>
</details>

<details>
<summary>See the certificate</summary>
<img src="https://github.com/user-attachments/assets/b081c33c-c8da-4494-8fe2-db23b1194063" alt="Certificate"></img>
</details>

## Reproducing Results

> <b>Note:</b>
> This project relies on the [now-deprecated .NET Interactive](https://github.com/dotnet/interactive).

### Data Collection

Clone the [StatsProjectTMNF](https://github.com/DarkMattrMaestro/StatsProjectTMNF) repository, or simply download the [`MainCodebase.ipynb`](https://github.com/DarkMattrMaestro/StatsProjectTMNF/blob/main/MainCodebase.ipynb) file and place it in a dedicated folder, and open the `MainCodebase.ipynb` file. This relies on the [now-deprecated .NET Interactive](https://github.com/dotnet/interactive), however some *potential* alternatives appear to be in-development. To open it in VSCode (my prefered choice), install the [Polyglot Notebooks](https://marketplace.visualstudio.com/items?itemName=ms-dotnettools.dotnet-interactive-vscode) extension and follow the setup steps (involving installing the correct version of the .NET SDK), reload VSCode, then set the notebook's kernel to to `.NET Interactive`.

Set `sampleSizePerTag` to however many samples you want per tag, with a minimum of 4. The list of tags collected are in the `UsableTag` enumerator. Additionally, you can add contact info into the user agent header at the line
```python
sharedClient.DefaultRequestHeaders.UserAgent.ParseAdd("SchoolStatsProject/1.0 (+https://github.com/DarkMattrMaestro/StatsProjectTMNF)");
```
for example, adding an email of `example@email.com` turning it into
```python
sharedClient.DefaultRequestHeaders.UserAgent.ParseAdd("SchoolStatsProject/1.0 (+https://github.com/DarkMattrMaestro/StatsProjectTMNF; example@email.com)");
```

When the notebook is configured as you deem fit, run it and wait for the collection program to finish. It may take a while depending on the number of samples you chose. A progress indicator is displayed at the bottom of the notebook.

Once the collection program is done (or even if it crashes partway), you should find a file named `{minRecords}recs-combined-flat-data.csv` where `{minRecords}` matches the value set at the top of the notebook. If you want the data from only one sampling attempt (and not a combination of all prior collections), you can search for `flat-replay-data.csv` in a directory structure that matches `{parent folder}/previous-data/{sampleSizePerTag}per-{minRecords}rep-{collection attempt #}/`. Copy the chosen CSV file to another location and rename it however you want for later use.

### Model Preparation & Rendering Figures

Clone the [stats-tmnf-quarto](https://github.com/DarkMattrMaestro/stats-tmnf-quarto) repository (the present repository). In the newly created folder, create a subfolder named `collected-data` and place your chosen CSV file into it. In `constants.py`, change `FILE_NAME` to point to the CSV file. For example, if my CSV file is named `5recs-combined-flat-data.csv`, then the line would become
```python
FILE_NAME = "./collected-data/5recs-combined-flat-data.csv"
```

Setup a Python virtual environment with the following commands (substitute `Activate.ps1` with `activate.bat` if you are using `cmd.exe`):
```
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

In the virtual environment, run `staty.py`, then run `shap_visuals.py` and `fig_model_stats.py`.
```
python3 staty.py
python3 shap_visuals.py
python3 fig_model_stats.py
```

Rendering the [quarto](https://quarto.org/docs/get-started/) file with `quarto render` will include all of the updated figures. See the generated `_manuscript` folder for the rendered PDF.

---

<details>
<summary>Français (cliquez pour élargir)</summary>

<a name="readme" href="https://darkmattrmaestro.github.io/stats-tmnf-quarto/Classification_de_toute_allure.pdf"><img src="https://github.com/user-attachments/assets/9a8b8bbf-81d8-4656-b379-b1296cd21b06" alt="Open_Graph_Classification_de_toute_allure"></img></a>

[Cliquez ici pour accéder au rapport.](https://darkmattrmaestro.github.io/stats-tmnf-quarto/Classification_de_toute_allure.pdf)

Les sections 4.2 et 4.3 sont les plus pertinentes.

## Informations générales

J'ai voulu mieux comprendre les étiquettes de circuits publiés sur [Trackmania Nations Forever Exchange](https://tmnf.exchange/). Ainsi, comme projet d'envergure pour mon cours de gestion de données, j'ai exploré les corrélations entre le cheminement d'une voiture et l'étiquette choisie pour le circuit. J'ai établi d'abord un modèle prédictif à but de prédire l'étiquette, suivi d'une analyse du modèle afin de tirer des conclusions.

Le modèle prédictif a une exactitude de 84,71%, ce qui est plutôt bon considérant qu'aucun renseignement du circuit, en soi, n'est fourni. En effet, le modèle n'a pas accès aux blocs qui composent les circuits.

## Affiche

J'ai créé une affiche pour le <a href="https://islp.ssc.ca/?lang=fr">concours national de l'ISLP</a> qui fait un sommaire de mon projet. J'admets que la qualité de mon affiche est assez basse ; j'ai traduit rapidement certaines sections pertinentes du rapport vers l'anglais (la langue de soumission requise par le concours). Toutefois, l'affiche a gagné la deuxième place.

<details>
<summary>Voir l'affiche</summary>
<img src="https://github.com/user-attachments/assets/48bf7334-d0a3-4e0a-b563-7730d1cfb67b" alt="Affiche"></img>
</details>

<details>
<summary>Voir le certificat</summary>
<img src="https://github.com/user-attachments/assets/b081c33c-c8da-4494-8fe2-db23b1194063" alt="Certificat"></img>
</details>


</details>
