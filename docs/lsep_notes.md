# Legal, Social, Ethical and Professional Issues — DermaRead

Use this as the basis for the LSEP section of your dissertation report.
Cite academic/legal sources in the final write-up (suggestions given).

---

## 1. Medical Device Regulation

DermaRead is explicitly a research prototype and is NOT classified as a medical device under:
- UK MDR 2002 (as amended post-Brexit)
- EU Medical Device Regulation (EU 2017/745)
- US FDA Software as a Medical Device (SaMD) guidance

For software to be classified as a medical device, it must be intended to diagnose, prevent, monitor, treat, or alleviate disease. DermaRead explicitly disclaims this intent via on-screen warnings. However, if this system were ever intended for clinical deployment, it would require CE/UKCA marking, clinical validation, and regulatory submission.

**Dissertation wording:** *"The system carries explicit on-screen disclaimers that predictions must not be used for diagnosis and that it is a research prototype, not a medical device. Under the UK Medical Devices Regulations 2002 and EU MDR 2017/745, software intended to aid clinical decision-making is regulated as a Class IIa or higher medical device. DermaRead does not meet the criteria for this classification as currently deployed; however, any attempt to use it in a clinical pathway would require formal regulatory approval and independent clinical validation."*

---

## 2. Data Protection and Image Consent

**Training data:** The primary training dataset (10,007 images) is a user-provided compilation of web-sourced skin images. No informed consent was obtained from the individuals depicted because the images were collected from publicly available sources. This raises a concern: medical images of identifiable body parts may constitute special category data under UK GDPR Article 9 (data concerning health). Strictly interpreted, processing such data requires explicit consent or a legitimate research basis under Article 89.

**External validation dataset:** Images were sourced from Kaggle datasets (various licences, predominantly CC-BY-NC) and supplementary web images used under fair-use principles for non-commercial academic research. No image data is stored beyond the project's local environment; no images are uploaded to external servers via the Gradio interface (local deployment only).

**Dissertation wording:** *"All images used in this project are sourced from publicly available datasets and used solely for non-commercial academic research. Under UK GDPR (Data Protection Act 2018), medical images may constitute special-category health data. No personally identifiable information accompanies any image, and the system does not retain uploaded images after inference. In a production deployment, a formal data protection impact assessment (DPIA) and ethics approval would be required."*

Cite: UK Data Protection Act 2018; EU GDPR Article 9; BCS Code of Conduct.

---

## 3. Algorithmic Fairness and Bias

The project directly addresses a known bias in dermatology AI: models trained predominantly on lighter-skinned patients perform worse on darker skin tones. This is documented in academic literature (Adamson & Smith, 2018; Daneshjou et al., 2021; Kinyanjui et al., 2019).

The ITA-based estimated skin-tone proxy is used to evaluate performance across estimated skin-tone groups. Limitations:
- ITA is distorted by lesion colour, lighting, and camera properties
- The training dataset contains only 9 verified dark-skin images in the external set
- Fairness claims based on this analysis must be stated as preliminary and exploratory, not conclusive

**Dissertation wording:** *"Algorithmic fairness in medical imaging has received significant academic attention following evidence that models trained on predominantly lighter-skinned datasets underperform on darker skin tones [cite Daneshjou et al., 2021]. This project attempts to address this through ITA-based skin-tone proxy analysis. However, the results must be interpreted cautiously: the training data lacks formal demographic annotation, ITA is an imperfect proxy, and the external dataset contains insufficient dark-skin representation for statistically reliable conclusions. This remains a significant limitation and represents an important direction for future work."*

Key citations:
- Daneshjou et al. (2021). *Disparities in dermatology AI performance on a diverse, curated clinical image set.* Science Advances.
- Adamson & Smith (2018). *Machine learning and health care disparities in dermatology.* JAMA Dermatology.
- Kinyanjui et al. (2019). *Estimating skin tone and the effects on classification performance in dermatology datasets.*

---

## 4. Clinical Responsibility and Patient Safety

A critical ethical issue with medical AI is the allocation of clinical responsibility. If a clinician uses an AI screening tool and misses a diagnosis, the legal and ethical question is: who is responsible?

- The clinician retains full legal and ethical responsibility under the GMC Good Medical Practice guidelines
- AI tools are advisory only and cannot override clinical judgement
- The "automation bias" risk — clinicians over-relying on AI predictions — is documented in the literature

DermaRead mitigates this through: mandatory disclaimer displayed at all times, explicit "not a medical device" labelling, confidence scores that communicate uncertainty, and Grad-CAM explanations that enable the user to critically evaluate the prediction.

**Dissertation wording:** *"A fundamental concern in clinical AI is automation bias — the tendency of practitioners to over-rely on algorithmic outputs at the expense of independent clinical reasoning [cite Parasuraman & Manzey, 2010]. DermaRead addresses this through prominent disclaimers, confidence scores, and Grad-CAM visualisations that make the model's reasoning transparent. Clinical responsibility under UK law and GMC guidelines remains with the qualified clinician; the system is designed as a screening aid only."*

---

## 5. Professional Standards — BCS Code of Conduct

As a computer science project, the BCS Code of Conduct (2022) applies:

- **Public interest:** The project aims to reduce diagnostic disparity for darker skin tones — a demonstrably public benefit
- **Competence:** The author is honest about the system's limitations (no clinical validation, distribution shift, imperfect fairness proxy)
- **Integrity:** Results are reported honestly; the finding that V1–V3 fail on real-world images is disclosed rather than suppressed
- **Maintaining professional competence:** The literature review demonstrates awareness of current state-of-the-art in dermatology AI

Cite: BCS (2022). *BCS Code of Conduct.* British Computer Society.

---

## 6. Environmental and Social Impact

- Training CNN models has a measurable carbon cost (estimated GPU hours: ~2-3 hours on RTX 3050)
- This is low relative to large language models but should be acknowledged
- Positive social impact: if validated, early psoriasis detection reduces diagnostic delay, which is associated with significant quality-of-life detriment and comorbidity risk (psoriatic arthritis affects ~30% of psoriasis patients)

---

## Key References for LSEP Section

1. Daneshjou et al. (2021). Disparities in dermatology AI. *Science Advances*, 7(31).
2. Adamson & Smith (2018). Machine learning and health care disparities. *JAMA Dermatology*, 156(11), 1317-1318.
3. Esteva et al. (2017). Dermatologist-level classification of skin cancer with deep neural networks. *Nature*, 542, 115-118.
4. Selvaraju et al. (2017). Grad-CAM: Visual explanations from deep networks. *ICCV 2017*.
5. He et al. (2016). Deep residual learning for image recognition. *CVPR 2016*.
6. BCS (2022). BCS Code of Conduct. British Computer Society.
7. UK GDPR / Data Protection Act 2018.
8. UK Medical Devices Regulations 2002.
9. Parasuraman & Manzey (2010). Complacency and bias in human use of automation. *Human Factors*, 52(3), 381-410.
10. Chardon et al. (1991). Skin colour typology and suntanning pathways. *International Journal of Cosmetic Science*, 13(4), 191-208. [ITA citation]
