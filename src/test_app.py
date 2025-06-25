from app import RefereeMatcher
import asyncio
import json

async def main():

    candidate_author_ids = [
        "https://openalex.org/authors/A5050993786", 
        "https://openalex.org/A5109035399",
        "https://openalex.org/authors/A5001986671",
        "https://openalex.org/authors/A5036198731",
        "https://openalex.org/authors/A5110066788",
        ]

    submitting_authors_ids = [
        'https://openalex.org/A5088589128', 
        'https://openalex.org/A5072058881', 
        'https://openalex.org/A5041807782',
        'https://openalex.org/A5087380109'
    ]

    matcher = RefereeMatcher(submitting_authors_ids)
    
    # You can test different abstracts; You'd need to change your topics_data variable accordingly
    abstract0 = 'Injection locking characteristics of oscillators are derived and a graphical analysis is presented that describes injection pulling in time and frequency domains. An identity obtained from phase and envelope equations is used to express the requisite oscillator nonlinearity and interpret phase noise reduction. The behavior of phase-locked oscillators under injection pulling is also formulated.'
    abstract1 = 'This study introduces a deep convolutional neural network model trained on retinal fundus images to automatically detect signs of diabetic retinopathy. Using a dataset of 35,000 annotated images, our model achieves an AUC of 0.96, outperforming existing computer-aided diagnostic tools. The approach demonstrates potential for large-scale automated screening.'
    abstract2 = 'A class-E RF power amplifier operating at 868 MHz is designed and implemented using a 65nm CMOS process for low-power IoT applications. The design achieves a power-added efficiency of 62% with a 10 dBm output power. A compact impedance-matching network and envelope shaping are employed to reduce power consumption.'
    abstract3 = 'We present a hybrid analog-digital beamforming architecture for mmWave massive MIMO systems using lens arrays and sparse channel estimation. Simulation results at 28 GHz show that our approach significantly reduces hardware complexity while achieving spectral efficiencies close to fully digital systems, making it ideal for 5G base stations.'
    abstract4 = 'We propose a transformer-based architecture for few-shot learning that leverages cross-attention and meta-representation adaptation. Our model outperforms existing benchmarks on the MiniImageNet and CIFAR-FS datasets with a 12% increase in classification accuracy, showing its effectiveness in low-data environments.'
    abstract5 = 'A computational study of unsteady flow around a rotating cylinder is conducted using Reynolds-Averaged Navier-Stokes equations. Results indicate a critical transition in vortex shedding frequency at specific Reynolds numbers, providing insight into drag reduction mechanisms for marine applications.'
    abstract6 = 'We investigate the quantum Hall effect in monolayer MoS₂ using low-temperature magnetotransport measurements. Observed plateaus at integer filling factors confirm the presence of a two-dimensional electron gas and support the spin-valley locking hypothesis in transition metal dichalcogenides.'
    abstract7 = 'A retrospective study of 1,200 breast cancer patients revealed that HER2 overexpression is correlated with poorer response to neoadjuvant chemotherapy. The use of trastuzumab improved disease-free survival by 35%, suggesting its continued utility in personalized oncology treatment protocols.'
    abstract8 = 'This study evaluates the seismic performance of reinforced concrete shear walls retrofitted with fiber-reinforced polymers (FRP). Using shake table tests, results show a 40% improvement in ductility and a significant delay in structural failure under simulated earthquake loading.'
    abstract9 = 'Through CRISPR/Cas9 editing, we knocked out the FOXP2 gene in mouse models to investigate its role in vocalization. The edited mice displayed altered ultrasonic vocal patterns, supporting the hypothesis that FOXP2 is critical for speech evolution and neurodevelopment.'
    abstract10 = 'Using satellite-based aerosol optical depth data, we show that particulate emissions in South Asia significantly contribute to regional monsoon suppression. Climate models incorporating this data predict a 12% reduction in seasonal rainfall by 2050 under current emission trajectories.'
    abstract11 = 'This paper explores the impact of framing effects on financial risk-taking among millennials. In a randomized experiment, subjects exposed to gain-framed messages were 24% more likely to invest in high-risk assets, highlighting the significance of behavioral nudges in policy design.'
    abstract12 = 'We report the synthesis of a NiFe-layered double hydroxide nanosheet catalyst for oxygen evolution in alkaline electrolyzers. The catalyst shows an overpotential of only 240 mV at 10 mA/cm² and maintains stability over 100 hours, marking a step toward efficient water splitting.'
    abstract13 = "Quantum many-body systems lie at the heart of modern condensed matter physics, quantum information science, and statistical mechanics. These systems consist of large ensembles of interacting particles whose collective quantum behavior gives rise to rich and often non-intuitive phenomena, such as quantum phase transitions, entanglement, and topological order. Understanding and simulating such systems remains a grand challenge due to the exponential complexity of their Hilbert space. Recent advances, including tensor network methods, quantum Monte Carlo, and machine learning-inspired approaches, have enabled significant progress in capturing the low-energy physics of various models. Moreover, experimental breakthroughs using ultracold atoms, superconducting qubits, and Rydberg atom arrays now allow precise control and observation of many-body dynamics in regimes once thought inaccessible. These developments are paving the way toward unraveling fundamental aspects of quantum matter and advancing technologies such as quantum simulation and computation."
    abstract14 = "Advanced photonic communication systems and optical network technologies enable ultra-fast, energy-efficient data transmission by harnessing the power of light. Photonic components such as lasers, modulators, and detectors support high-speed signal processing, while modern optical networks—including SDN-enabled and space-division multiplexed systems—ensure scalable, flexible connectivity. Together, these innovations form the backbone of next-generation infrastructure for data centers, cloud computing, and 5G/6G networks."
    abstract15 = "This paper explores recent innovations in advanced wireless communication techniques with a focus on the enabling role of cutting-edge Phase-Locked Loop (PLL) and Voltage-Controlled Oscillator (VCO) technologies. We examine how improvements in PLL and VCO design contribute to enhanced frequency synthesis, phase noise reduction, and overall signal integrity, which are critical for next-generation high-speed and low-power wireless systems. The synergy between these circuit-level advancements and modern communication architectures is analyzed, highlighting their impact on spectral efficiency, reliability, and scalability in emerging wireless applications."

    topics_data = [
        {"topic_id": "https://openalex.org/T10125", "topic_name": "Advanced Wireless Communication Technique"},
        {"topic_id": "https://openalex.org/T11417", "topic_name": "Advancements in PLL and VCO Technologies"},
        # ... possibly more, but we want just the top 3
    ]

    all_top_works = await matcher.fetch_all_topic_works(topics_data)

    sorted_works_by_relevance = await matcher.sort_works_by_relevance(all_top_works, abstract15)
    top_referees = await matcher.get_top_referees(sorted_works_by_relevance, from_year=2016, min_citations=15, max_citations=80)
    
    top_referee_works = matcher.extract_batched_works(top_referees)
    rej_works = await matcher.reject_irrelevant_works_from_referees(top_referee_works, abstract=abstract15)
    updated_referees = matcher.apply_work_rejections(top_referees, rej_works)

    # Save to a file
    with open("top_referees_pre.json", "w", encoding="utf-8") as f:
        json.dump(top_referees, f, ensure_ascii=False, indent=2)

    with open("rejected_works.json", "w", encoding="utf-8") as f:
        json.dump(rej_works, f, ensure_ascii=False, indent=2)

    with open("top_referees_post.json", "w", encoding="utf-8") as f:
        json.dump(updated_referees, f, ensure_ascii=False, indent=2)

    pub_profiles = matcher.build_pub_history_from_referees(updated_referees)  

    # Optionally save to JSON
    with open("pub_referees_profiles.json", "w", encoding="utf-8") as f:
        json.dump(pub_profiles, f, ensure_ascii=False, indent=2)  

# Run main
asyncio.run(main())