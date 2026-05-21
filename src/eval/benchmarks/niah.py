"""Needle-in-a-Haystack benchmark.

Generates a synthetic long document (haystack) and embeds a short, distinctive
fact (needle) at a precise depth. Tests whether the model can retrieve the needle
across varying context lengths and depths.
"""
from __future__ import annotations
import random
from dataclasses import dataclass

from ..metrics import needle_hit

# 20 topically-varied paragraphs used to construct haystacks.
_HAYSTACK_PARAS = [
    "The annual migration of monarch butterflies represents one of nature's most remarkable phenomena. Each autumn, millions of butterflies travel thousands of miles from Canada and the northern United States to overwinter in the mountain forests of central Mexico. Scientists have spent decades studying how these insects navigate using Earth's magnetic field and the position of the sun, yet many aspects of this journey remain incompletely understood.",
    "In materials science, researchers have been investigating the properties of two-dimensional materials since the isolation of graphene in 2004. This single-layer arrangement of carbon atoms exhibits extraordinary strength, electrical conductivity, and flexibility. Potential applications include next-generation electronics, biomedical devices, and structural composites used in aerospace engineering.",
    "The history of the printing press and its impact on the spread of knowledge across medieval Europe is well documented. Gutenberg's innovation allowed texts to be reproduced at unprecedented scale, enabling the rapid distribution of religious, scientific, and philosophical works. The standardization of languages and alphabets followed naturally from this new capacity for mass communication.",
    "Coral reefs cover less than one percent of the ocean floor yet support roughly twenty-five percent of all marine species. Rising ocean temperatures and acidification, both consequences of increased atmospheric carbon dioxide, have led to widespread bleaching events in reef ecosystems around the world. Restoration projects using heat-resistant coral fragments offer one avenue for preserving these critically important habitats.",
    "The development of the transistor at Bell Laboratories in 1947 fundamentally altered the trajectory of computing technology. By replacing bulky vacuum tubes with small solid-state switches, engineers could build ever more compact and reliable circuits. The subsequent miniaturization of transistors onto integrated circuits followed Moore's Law for several decades before physical limits began to constrain further scaling.",
    "Agricultural civilization emerged independently in several regions of the world between ten thousand and five thousand years ago. The domestication of wheat and barley in the Fertile Crescent, rice in the Yangtze River valley, and maize in Mesoamerica each enabled population densities far beyond what hunter-gatherer lifestyles could sustain. Surplus food production supported the emergence of cities, writing systems, and specialized labor.",
    "The deep ocean, defined as water below two hundred meters, remains one of the least explored environments on Earth. Extreme pressure, near-freezing temperatures, and total darkness characterize this realm, yet it harbors remarkable biodiversity including bioluminescent organisms and chemosynthetic bacteria at hydrothermal vents. Submersible technology has only recently begun to reveal the full scope of life in this biome.",
    "Classical conditioning, first described by Ivan Pavlov in his experiments with dogs at the turn of the twentieth century, established the foundational principles of behavioral learning theory. The pairing of a neutral stimulus with an unconditioned stimulus leads the subject to produce a conditioned response to the neutral stimulus alone. This mechanism underlies a wide range of learned behaviors across vertebrate and invertebrate species.",
    "Wind energy has grown from a marginal contribution to the global electricity supply to a significant source of renewable power over the past three decades. Advances in turbine blade aerodynamics, generator efficiency, and offshore foundation engineering have driven down the levelized cost of wind electricity substantially. Grid integration challenges, including intermittency and transmission infrastructure, remain active areas of policy development.",
    "The structure of DNA was elucidated by Watson and Crick in 1953, building on X-ray diffraction data produced by Rosalind Franklin and others. The double helix model explained how genetic information could be stored in the sequence of nucleotide base pairs and faithfully replicated during cell division. This discovery opened the door to the modern era of molecular biology and eventually to technologies such as genetic engineering and gene therapy.",
    "Ancient trade routes connected distant civilizations and facilitated the exchange not only of goods but also of ideas, diseases, and technologies. The Silk Road linked China to the Mediterranean world, carrying silk, spices, glassware, and precious metals across deserts and mountain ranges. The diffusion of Buddhism, Islam, and other religious traditions followed these same commercial corridors across central Asia.",
    "The human immune system distinguishes between self and non-self through a complex array of molecular recognition mechanisms. B cells and T cells carry receptors capable of binding to specific antigens on the surface of pathogens or infected host cells. When these receptors engage their targets, a cascade of signaling events culminates in the destruction of the invader and the formation of immunological memory.",
    "Plate tectonics describes the movement of large sections of Earth's lithosphere over the partially molten asthenosphere below. The collision of plates builds mountain ranges, the separation of plates creates ocean basins, and subduction zones recycle oceanic crust back into the mantle. Earthquake and volcanic activity are concentrated along plate boundaries where these dynamic interactions are most intense.",
    "The philosophy of language grapples with questions about how words acquire meaning and how sentences convey information about reality. Philosophers including Frege, Russell, and Wittgenstein developed theories of reference, truth conditions, and language games that continue to influence both linguistics and analytic philosophy. The relationship between language, thought, and the world remains one of the central puzzles of modern philosophy.",
    "Space telescopes have transformed our understanding of the universe by observing wavelengths that do not penetrate Earth's atmosphere. The Hubble Space Telescope revealed the expansion rate of the universe, the existence of supermassive black holes in galactic centers, and the remarkable diversity of galaxy morphologies across cosmic time. Subsequent observatories operating in infrared, ultraviolet, and X-ray wavelengths extended this observational reach further still.",
    "Fermentation has been employed by human cultures for thousands of years to preserve food and produce alcoholic beverages. The metabolic activity of yeast and bacteria transforms sugars into ethanol, lactic acid, carbon dioxide, and various flavor compounds depending on the microorganism and substrate involved. Modern biotechnology has harnessed fermentation for the production of pharmaceuticals, biofuels, and industrial enzymes at commercial scale.",
    "Socioeconomic inequality and its consequences for health outcomes have been documented across a wide range of societies and historical periods. Access to nutritious food, clean water, adequate housing, and medical care correlates strongly with income and wealth. Public health interventions targeting structural determinants of health rather than individual behaviors have shown promise in reducing these disparities at the population level.",
    "The physics of fluid dynamics governs phenomena from the circulation of blood in arteries to the formation of weather systems in the atmosphere. The Navier-Stokes equations describe the motion of viscous fluids but remain analytically unsolved in the general turbulent case, representing one of the outstanding problems of classical physics. Computational fluid dynamics provides numerical approximations that enable engineering design across aerospace and biomedicine.",
    "Languages evolve through processes of sound change, grammatical restructuring, lexical borrowing, and contact with neighboring speech communities. Historical linguists reconstruct proto-languages by comparing systematic correspondences among related languages and applying the comparative method. The family tree model and the wave model offer complementary perspectives on how linguistic variation spreads across geographic and social space.",
    "Photosynthesis converts light energy from the sun into chemical energy stored in carbohydrates, releasing oxygen as a byproduct. In plants, algae, and cyanobacteria, this process occurs in two main stages: the light-dependent reactions that generate ATP and NADPH, and the Calvin cycle that uses these molecules to fix carbon dioxide into sugars. The efficiency of this pathway limits agricultural productivity and has motivated efforts to engineer improved photosynthetic routes.",
]

# Each tuple: (needle sentence, answer value, question).
# Answers are distinctive 5-digit numbers unlikely to appear by chance.
_NEEDLE_FACTS = [
    (
        "Please make note: the secret number recorded in this document is 74829.",
        "74829",
        "What is the secret number recorded in this document?",
    ),
    (
        "Important: the special code embedded in this passage is 39157.",
        "39157",
        "What is the special code embedded in this passage?",
    ),
    (
        "Take note that the hidden value stored in this text is 62483.",
        "62483",
        "What is the hidden value stored in this text?",
    ),
    (
        "Remember: the unique identifier inserted into this document is 85614.",
        "85614",
        "What is the unique identifier inserted into this document?",
    ),
    (
        "A key fact has been placed here: the reference number is 47293.",
        "47293",
        "What is the reference number placed in this document?",
    ),
]


@dataclass
class NIAHSample:
    context: str
    question: str
    answer: str
    context_length_target: int
    depth: float
    trial: int


def generate_samples(
    context_lengths: list[int] | None = None,
    depths: list[float] | None = None,
    n_trials: int = 3,
    rng: random.Random | None = None,
) -> list[NIAHSample]:
    if context_lengths is None:
        context_lengths = [1000, 2000, 4000, 8000, 16000]
    if depths is None:
        depths = [0.1, 0.25, 0.5, 0.75, 0.9]
    if rng is None:
        rng = random.Random(42)

    samples: list[NIAHSample] = []
    for length in context_lengths:
        for depth in depths:
            for trial in range(n_trials):
                needle_text, answer, question = _NEEDLE_FACTS[trial % len(_NEEDLE_FACTS)]
                paragraphs = _build_paragraphs(length, rng)
                context = _insert_needle(paragraphs, needle_text, depth)
                samples.append(NIAHSample(
                    context=context,
                    question=question,
                    answer=answer,
                    context_length_target=length,
                    depth=depth,
                    trial=trial,
                ))
    return samples


def format_prompt(sample: NIAHSample) -> str:
    return (
        f"{sample.context}\n\n"
        f"Question: {sample.question}\n\n"
        "Answer:"
    )


def score_sample(sample: NIAHSample, prediction: str) -> dict:
    return {
        "context_length_target": sample.context_length_target,
        "depth": sample.depth,
        "trial": sample.trial,
        "answer": sample.answer,
        "prediction": prediction,
        "hit": needle_hit(prediction, sample.answer),
    }


def _build_paragraphs(target_tokens: int, rng: random.Random) -> list[str]:
    # ~0.75 words per token for Llama; each paragraph ~60 words
    target_words = int(target_tokens * 0.75)
    pool = _HAYSTACK_PARAS[:]
    rng.shuffle(pool)

    paragraphs: list[str] = []
    word_count = 0
    idx = 0
    while word_count < target_words:
        para = pool[idx % len(pool)]
        idx += 1
        paragraphs.append(para)
        word_count += len(para.split())
    return paragraphs


def _insert_needle(paragraphs: list[str], needle: str, depth: float) -> str:
    idx = max(0, min(len(paragraphs) - 1, int(len(paragraphs) * depth)))
    result = paragraphs[:idx] + [needle] + paragraphs[idx:]
    return "\n\n".join(result)
