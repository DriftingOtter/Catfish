# Catfish

> A finance trend forecasting & trading multimodel framework.

---

## Trading Strategy

Catfish is a distributed, horizontally scalable trading system with strategy-partitioned agents, self-regulating execution logic, and local/global performance caching for validated order execution across multiple trading horizons.

```mermaid
flowchart TD
    DB["Digital Broker"]

    DB --> F1["Feature #1"]
    DB --> FDOTS["···"]
    DB --> FN["Feature #n"]

    F1 --> M1["Model #1"]
    F1 --> M2["Model #2"]
    F1 --> MDOTS["···"]
    F1 --> MN["Model #n"]

    FN --> FNDOTS["···"]

    style DB     fill:#1e3a5f,color:#fff,stroke:#1e3a5f
    style F1     fill:#2d6a9f,color:#fff,stroke:#2d6a9f
    style FN     fill:#2d6a9f,color:#fff,stroke:#2d6a9f
    style M1     fill:#3a8fbf,color:#fff,stroke:#3a8fbf
    style M2     fill:#3a8fbf,color:#fff,stroke:#3a8fbf
    style MN     fill:#3a8fbf,color:#fff,stroke:#3a8fbf
    style FDOTS  fill:none,color:#666,stroke:none
    style MDOTS  fill:none,color:#666,stroke:none
    style FNDOTS fill:none,color:#666,stroke:none
```

The goal of the **Digital Broker** is to analyse the scale and validity of each model's predictions, weigh the impact of each model, and reason towards a potential buy or sell order.

---

## Multi-Broker Moots (MBMs)

Catfish runs multiple Digital Brokers in parallel, each dedicated to a distinct trading strategy with its own hierarchy of features and underlying models. Rather than operating in isolation, the brokers are deployed simultaneously across many relevant stock options in relation to the primary change order, with each broker working to validate or disprove the proposed actions of the others before any official request is committed. This mutual scrutiny ensures that no single strategy can unilaterally drive an execution, and that the system as a whole remains robust to the failure or overconfidence of any one model ensemble. The outputs of all active brokers are then consolidated into a single change order node, which is forwarded to the Finance Agent for final adjudication.

```mermaid
flowchart TD
    subgraph DB1["Digital Broker #1 · Strategy #1"]
        F1a["Feature #1"]  --> M1a["Model #1"]
        F1a                --> M1b["Model #2"]
        F1a                --> M1d["···"]
        F1a                --> M1n["Model #n"]

        F1n1["Feature #n"] --> FN1d["···"]
    end

    subgraph DB2["Digital Broker #2 · Strategy #2"]
        F2a["Feature #1"]  --> M2a["Model #1"]
        F2a                --> M2b["Model #2"]
        F2a                --> M2d["···"]
        F2a                --> M2n["Model #n"]

        F2n1["Feature #n"] --> FN2d["···"]
    end

    subgraph DBN["Digital Broker #n · Strategy #n"]
        FNa["Feature #1"]  --> MNa["Model #1"]
        FNa                --> MNb["Model #2"]
        FNa                --> MNd["···"]
        FNa                --> MNn["Model #n"]

        FNn1["Feature #n"] --> FNNd["···"]
    end

    DB1 --> PO["Proposed Order"]
    DB2 --> PO
    DBN --> PO

    PO --> CO["Change Order (Buy / Sell)"]
    CO --> FA["Finance Agent"]

    %% Styling
    style PO fill:#2d2d4e,color:#bbb,stroke:#555
    style CO fill:#4a235a,color:#fff,stroke:#4a235a
    style FA fill:#1a5c3a,color:#fff,stroke:#1a5c3a

    %% Match ellipses to surrounding feature/model layer colors
    style M1d fill:#3a8fbf,color:#fff,stroke:#3a8fbf
    style M2d fill:#3a8fbf,color:#fff,stroke:#3a8fbf
    style MNd fill:#3a8fbf,color:#fff,stroke:#3a8fbf

    style FN1d fill:#3a8fbf,color:#fff,stroke:#3a8fbf
    style FN2d fill:#3a8fbf,color:#fff,stroke:#3a8fbf
    style FNNd fill:#3a8fbf,color:#fff,stroke:#3a8fbf
```

> **Note:** The implementation of MBMs is user-dependent, but is currently focused on **NASDAQ index funds**.

---

## Exposure × Financial Analysis Automation

The Finance Agent serves as the final arbiter across all incoming change orders, receiving a proposed order alongside a confidence signal from every active Digital Broker simultaneously. Rather than evaluating each proposed order in isolation, the agent maintains a persistent memory of the historical performance of each broker, drawing on records of previous orders, cumulative profit and loss, win rate, and a broader suite of statistical measures, including Sharpe ratio and maximum drawdown, in order to construct a current trust weighting for each broker. 

```mermaid
flowchart TD
    DB1["Digital Broker #1 Strategy #1"]
    DB2["Digital Broker #2 Strategy #2"]
    DBN["Digital Broker #n Strategy #n"]

    DBN --> POC["Proposed Order + Confidence"]
    DB2 --> POC
    DB1 --> POC

    POC --> FA["Finance Agent"]

    subgraph Mem["Agent Memory"]
        PO["Previous Orders\n(per broker)"]
        PL["P&L History"]
        WR["Win Rate"]
        SA["Statistical Metrics\n(Sharpe · Max Drawdown · etc)"]
    end

    FA  --> UPD["Update Memory"]
    UPD --> Mem
    Mem --> READ["Read History"]
    READ --> FA

    FA --> HT["High Trust"]
    HT --> EX["✓ Execute Order"]

    FA --> LT["Low Trust"]
    LT --> HO["✗ Hold / Reject"]

    %% Styling (core nodes)
    style FA   fill:#1a5c3a,color:#fff,stroke:#1a5c3a
    style Mem  fill:#1c2e40,color:#fff,stroke:#2d6a9f
    style EX   fill:#1e4d2b,color:#fff,stroke:#1e4d2b
    style HO   fill:#4d1e1e,color:#fff,stroke:#4d1e1e
    style POC  fill:#2d2d4e,color:#bbb,stroke:#555
    style UPD  fill:#2d2d4e,color:#bbb,stroke:#555
    style READ fill:#2d2d4e,color:#bbb,stroke:#555
    style HT   fill:#2d2d4e,color:#bbb,stroke:#555
    style LT   fill:#2d2d4e,color:#bbb,stroke:#555

    %% Memory internal nodes (consistent styling)
    style PO fill:#1c2e40,color:#ccc,stroke:#2d6a9f
    style PL fill:#1c2e40,color:#ccc,stroke:#2d6a9f
    style WR fill:#1c2e40,color:#ccc,stroke:#2d6a9f
    style SA fill:#1c2e40,color:#ccc,stroke:#2d6a9f
```

The agent thus weighs not only the content of each proposed order but also the established track record of the strategy that produced it; a broker with a strong and consistent history of profitable decisions will carry greater influence over the final outcome than one whose recent performance has been poor or erratic. Orders whose weighted trust clears the agent's confidence threshold are forwarded for execution, whilst those that fall below are held or rejected outright. The agent's memory is updated after every resolved order, ensuring that trust weightings remain responsive to each broker's most recent behaviour rather than relying solely on static historical averages.

To further reduce execution risk, fundamental statistical testing is applied to any outgoing buy order to confirm its financial stability with respect to both the **value** and **flexibility** of the current portfolio prior to settlement.
