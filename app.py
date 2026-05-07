"""Flask entrypoint for the MRI classification demo application.

This file defines the pages, connects buttons/forms to the backend helpers,
and passes prebuilt bundles to the templates.
"""

from flask import Flask, flash, redirect, render_template, request, url_for

from mri_interface import (
    CLASSES,
    build_cah_cluster_bundle,
    build_data_demo_bundle,
    build_kmeans_binary_class_bundle,
    get_cah_cluster_cards,
    get_cah_cluster_overview,
    build_kmeans_class_bundle,
    get_kmeans_binary_artifacts,
    predict_kmeans_binary_image,
    run_cah_pipeline,
    run_kmeans_pipeline,
)


# Static educational content shown on the diagnosis information pages.
DIAGNOSIS_CONTENT = {
    "no_tumor": {
        "page_title": "Resultat Normal",
        "status": "Resultat normal",
        "badge": "No Tumor",
        "image": "img/no-tumor-diagnosis.png",
        "heading": "Aucune anomalie detectee",
        "summary": (
            "L'analyse par intelligence artificielle n'a identifie aucun signe de tumeur cerebrale "
            "dans les images IRM. C'est un resultat rassurant, mais il doit toujours etre confirme "
            "par un medecin."
        ),
        "stats": [
            {"label": "Detection", "value": "0 anomalie", "caption": "Aucune masse compatible avec une tumeur visible."},
            {"label": "Interpretation", "value": "Resultat negatif", "caption": "Aucun profil tumoral reconnu par l'algorithme."},
            {"label": "Suivi", "value": "Medical", "caption": "A confirmer avec un professionnel de sante."},
        ],
        "meaning": [
            "Un resultat negatif signifie que l'IA n'a detecte aucune masse, aucune zone anormale correspondant aux profils connus des tumeurs cerebrales.",
            "Cela ne veut pas dire que tout va bien dans l'absolu, mais c'est un indicateur solide que le cerveau ne presente pas d'anomalie tumorale visible sur cette IRM.",
            "Certains symptomes comme les maux de tete persistants, les vertiges ou les troubles visuels peuvent avoir d'autres origines que les tumeurs.",
        ],
        "symptoms": [
            "Si des symptomes persistent, ils peuvent relever de causes vasculaires, inflammatoires, hormonales ou du stress.",
            "Un medecin pourra orienter vers les examens complementaires adaptes si la clinique le justifie.",
        ],
        "treatments": ["Partager avec votre medecin", "Signaler tout symptome persistant", "Suivi regulier conseille"],
        "treatment_note": "Ce resultat est une premiere analyse automatisee et ne remplace pas un diagnostic medical complet.",
        "takeaways": [
            "Partagez ce resultat avec votre medecin.",
            "N'ignorez pas un symptome inquietant meme avec un resultat rassurant.",
            "Un suivi clinique reste recommande si les signes persistent.",
        ],
    },
    "glioma": {
        "page_title": "Gliome",
        "status": "Anomalie detectee",
        "badge": "Glioma",
        "image": "img/glioma-diagnosis.png",
        "heading": "Le gliome est une tumeur des cellules gliales",
        "summary": (
            "Le gliome prend naissance dans les cellules de soutien du cerveau. C'est la tumeur "
            "cerebrale primaire la plus courante, avec des formes allant du lent au tres agressif."
        ),
        "stats": [
            {"label": "Frequence", "value": "~30 %", "caption": "Des tumeurs cerebrales primaires."},
            {"label": "Grades", "value": "I a IV", "caption": "Du plus lent au plus agressif."},
            {"label": "Age typique", "value": "45-65 ans", "caption": "Age frequent au diagnostic."},
        ],
        "meaning": [
            "Les gliomes sont classes en quatre grades selon leur agressivite. Les grades I et II evoluent lentement, tandis que les grades III et IV sont plus agressifs.",
            "Le glioblastome correspond au grade IV, la forme la plus severe. Le grade exact ne peut pas etre confirme par IRM seule et demande une biopsie.",
        ],
        "symptoms": [
            "Les symptomes dependent de la zone atteinte: maux de tete matinaux, crises d'epilepsie nouvelles, troubles du langage, faiblesse d'un cote ou changement de comportement.",
        ],
        "treatments": ["Chirurgie", "Radiotherapie", "Chimiotherapie", "Therapies ciblees", "Immunotherapie"],
        "treatment_note": "Une detection precoce ameliore les perspectives de traitement. Une evaluation neuro-oncologique reste essentielle.",
        "takeaways": [
            "Bas grade: environ 40 % des cas.",
            "Haut grade: environ 60 % des cas.",
            "Un neurochirurgien ou neuro-oncologue doit confirmer le plan de prise en charge.",
        ],
    },
    "meningioma": {
        "page_title": "Meningiome",
        "status": "Anomalie detectee",
        "badge": "Meningioma",
        "image": "img/meningioma-diagnosis.png",
        "heading": "Le meningiome se developpe dans les meninges",
        "summary": (
            "Le meningiome est la tumeur cerebrale la plus frequente. Dans plus de 90 % des cas, "
            "il est benin et evolue lentement."
        ),
        "stats": [
            {"label": "Frequence", "value": "~36 %", "caption": "Des tumeurs cerebrales."},
            {"label": "Benignite", "value": "> 90 %", "caption": "Des cas sont non malins."},
            {"label": "Croissance", "value": "Tres lente", "caption": "Parfois silencieuse pendant des annees."},
        ],
        "meaning": [
            "Le meningiome pousse a l'exterieur du tissu cerebral, ce qui le rend souvent plus accessible a la chirurgie.",
            "Une forme atypique ou maligne existe mais reste rare. La localisation reste un facteur cle pour juger le risque reel.",
        ],
        "symptoms": [
            "Les signes possibles sont des maux de tete progressifs, troubles visuels, baisse de l'audition ou faiblesse musculaire selon la zone comprimee.",
            "Beaucoup de patients n'ont aucun symptome au moment du diagnostic.",
        ],
        "treatments": ["Surveillance IRM reguliere", "Chirurgie", "Radiochirurgie stereotaxique"],
        "treatment_note": "Une petite tumeur pres d'une zone critique peut etre plus importante qu'une grosse tumeur situee dans une zone accessible.",
        "takeaways": [
            "Benin grade I: environ 90 %.",
            "Atypique ou malin grades II-III: environ 10 %.",
            "La decision depend de la taille, la localisation et les symptomes.",
        ],
    },
    "pituitary": {
        "page_title": "Pituitary Tumor",
        "status": "Anomalie detectee",
        "badge": "Pituitary",
        "image": "img/pituitary-diagnosis.png",
        "heading": "La tumeur hypophysaire touche la glande endocrine centrale",
        "summary": (
            "Cette tumeur se developpe sur l'hypophyse, une petite glande a la base du cerveau. "
            "Elle est presque toujours benigne, mais peut perturber fortement l'equilibre hormonal."
        ),
        "stats": [
            {"label": "Frequence", "value": "~15 %", "caption": "Des tumeurs cerebrales."},
            {"label": "Nature", "value": "> 95 %", "caption": "Des cas sont benins."},
            {"label": "Taille", "value": "Micro / Macro", "caption": "Inferieure ou superieure a 1 cm."},
        ],
        "meaning": [
            "Le principal enjeu n'est pas la malignite, mais l'impact sur la production hormonale.",
            "Certaines tumeurs secretent trop d'hormones, d'autres compriment l'hypophyse ou le nerf optique.",
        ],
        "symptoms": [
            "Fatigue chronique, prise ou perte de poids, irregularites menstruelles, baisse de libido, maux de tete centraux ou perte progressive de la vision peripherique.",
        ],
        "treatments": ["Medicaments", "Chirurgie endonasale", "Radiotherapie"],
        "treatment_note": "Un endocrinologue et un neurochirurgien travaillent souvent ensemble pour la prise en charge complete.",
        "takeaways": [
            "Benigne: plus de 95 % des cas.",
            "Maligne: moins de 1 % des cas.",
            "L'impact hormonal peut etre subtil mais progressif.",
        ],
    },
}

app = Flask(__name__)
app.config["SECRET_KEY"] = "brain-tumor-mri-demo"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
@app.route("/")
def landing():
    """Render the landing page."""
    return render_template("landing.html")


@app.route("/learn")
def learn():
    """Render the educational anatomy page."""
    return render_template("learn.html")


@app.route("/diagnosis/<diagnosis_slug>")
def diagnosis_page(diagnosis_slug):
    info = DIAGNOSIS_CONTENT.get(diagnosis_slug)
    if not info:
        return redirect(url_for("learn"))
    return render_template(
        "diagnosis_info.html",
        page_title=info["page_title"],
        info=info,
    )


@app.route("/data")
def data_page():
    try:
        bundle = build_data_demo_bundle()
    except Exception as exc:
        flash(f"Affichage Data impossible: {exc}", "error")
        return redirect(url_for("landing"))

    return render_template(
        "data.html",
        page_title="Data / Image Processing",
        bundle=bundle,
    )


@app.route("/dashboard")
def index():
    return render_template(
        "index.html",
        classes=CLASSES,
    )


@app.route("/method/<method_name>")
def method_page(method_name):
    if method_name == "kmeans":
        try:
            bundle = run_kmeans_pipeline()
        except Exception as exc:
            flash(f"Execution CAMU impossible: {exc}", "error")
            return redirect(url_for("landing"))
        return render_template(
            "method.html",
            method_name=method_name,
            page_title="K-Means",
            classes=CLASSES,
            bundle=bundle,
            prediction=None,
        )

    if method_name == "kmeans2":
        try:
            bundle = get_kmeans_binary_artifacts()
        except Exception as exc:
            flash(f"Execution K-Means k=2 impossible: {exc}", "error")
            return redirect(url_for("landing"))
        return render_template(
            "method.html",
            method_name=method_name,
            page_title="K-Means Binary (k = 2)",
            classes=["tumor", "no_tumor"],
            bundle=bundle,
            prediction=None,
        )

    if method_name == "cah":
        try:
            bundle = run_cah_pipeline()
            bundle["cluster_cards"] = get_cah_cluster_cards()
            bundle.update(get_cah_cluster_overview())
        except Exception as exc:
            flash(f"Execution CAH impossible: {exc}", "error")
            return redirect(url_for("landing"))
        return render_template(
            "method.html",
            method_name=method_name,
            page_title="CAH / Hierarchical Classification",
            classes=CLASSES,
            bundle=bundle,
            prediction=None,
        )

    return redirect(url_for("landing"))


@app.route("/method/kmeans/<class_name>")
def kmeans_class_page(class_name):
    try:
        bundle = build_kmeans_class_bundle(class_name)
    except Exception as exc:
        flash(f"Affichage de la classe impossible: {exc}", "error")
        return redirect(url_for("method_page", method_name="kmeans"))

    return render_template(
        "kmeans_class.html",
        page_title=f"K-Means - {bundle['class_label']}",
        classes=CLASSES,
        bundle=bundle,
    )


@app.route("/method/kmeans2/class/<class_name>")
def kmeans_binary_class_page(class_name):
    try:
        bundle = build_kmeans_binary_class_bundle(class_name)
    except Exception as exc:
        flash(f"Affichage de la classe binaire impossible: {exc}", "error")
        return redirect(url_for("method_page", method_name="kmeans2"))

    return render_template(
        "kmeans_binary_class.html",
        page_title=f"K-Means k=2 - {bundle['class_label']}",
        bundle=bundle,
    )


@app.route("/method/cah/cluster/<int:cluster_id>")
def cah_cluster_page(cluster_id):
    try:
        bundle = build_cah_cluster_bundle(cluster_id)
    except Exception as exc:
        flash(f"Affichage du cluster CAH impossible: {exc}", "error")
        return redirect(url_for("method_page", method_name="cah"))

    return render_template(
        "cah_cluster.html",
        page_title=f"CAH - {bundle['cluster_label']}",
        classes=CLASSES,
        bundle=bundle,
    )


@app.route("/kmeans2/predict", methods=["POST"])
def kmeans2_predict():
    file = request.files.get("image")
    if not file or not file.filename:
        flash("Veuillez choisir une image MRI avant de lancer la prediction.", "error")
        return redirect(url_for("method_page", method_name="kmeans2"))

    try:
        prediction = predict_kmeans_binary_image(file)
        bundle = get_kmeans_binary_artifacts()
    except ValueError as exc:
        flash(str(exc), "error")
        return redirect(url_for("method_page", method_name="kmeans2"))
    except Exception as exc:
        flash(f"Prediction K-Means k=2 impossible: {exc}", "error")
        return redirect(url_for("method_page", method_name="kmeans2"))

    return render_template(
        "method.html",
        method_name="kmeans2",
        page_title="K-Means Binary (k = 2)",
        classes=["tumor", "no_tumor"],
        bundle=bundle,
        prediction=prediction,
    )


if __name__ == "__main__":
    app.run(debug=True)
