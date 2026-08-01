const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, ImageRun, TabStopType, PageBreak
} = require("docx");

const FONT = "Times New Roman";

function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 160, line: 300 },
    alignment: opts.align || AlignmentType.JUSTIFIED,
    children: [new TextRun({ text, font: FONT, size: 24, italics: !!opts.italics, bold: !!opts.bold })],
  });
}

function heading(text, level) {
  return new Paragraph({
    heading: level,
    spacing: { before: 260, after: 140 },
    children: [new TextRun({ text, font: FONT, bold: true })],
  });
}

function caption(text) {
  return new Paragraph({
    spacing: { after: 200 },
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text, font: FONT, size: 20, bold: true })],
  });
}

function refPara(text) {
  return new Paragraph({
    spacing: { after: 120, line: 276 },
    indent: { left: 360, hanging: 360 },
    alignment: AlignmentType.JUSTIFIED,
    children: [new TextRun({ text, font: FONT, size: 22 })],
  });
}

function eq(text, num) {
  return new Paragraph({
    spacing: { before: 120, after: 120 },
    tabStops: [{ type: TabStopType.RIGHT, position: 9020 }],
    alignment: AlignmentType.LEFT,
    children: [
      new TextRun({ text: "\t" + text, font: "Cambria Math", size: 24 }),
      new TextRun({ text: `\t(${num})`, font: FONT, size: 22 }),
    ],
  });
}

function eqNote(text) {
  return new Paragraph({
    spacing: { after: 160, line: 280 },
    alignment: AlignmentType.JUSTIFIED,
    children: [new TextRun({ text, font: FONT, size: 22, italics: true })],
  });
}

function cellText(text, opts = {}) {
  return new TableCell({
    width: { size: opts.width || 1600, type: WidthType.DXA },
    shading: opts.header ? { type: ShadingType.CLEAR, fill: "D9E2F3" } : undefined,
    verticalAlign: "center",
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text, font: FONT, size: 20, bold: !!opts.header })],
    })],
  });
}

function dataTable(headerRow, rows, colWidths) {
  return new Table({
    width: { size: colWidths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [
      new TableRow({
        tableHeader: true,
        children: headerRow.map((h, i) => cellText(h, { header: true, width: colWidths[i] })),
      }),
      ...rows.map(r => new TableRow({
        children: r.map((c, i) => cellText(String(c), { width: colWidths[i] })),
      })),
    ],
  });
}

function figure(imgPath, capText, widthPx = 380, heightPx = 300) {
  const buf = fs.readFileSync(imgPath);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 80 },
      children: [new ImageRun({ data: buf, transformation: { width: widthPx, height: heightPx }, type: "png" })],
    }),
    caption(capText),
  ];
}

const children = [];

// ================= Title block =================
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 120 },
  children: [new TextRun({
    text: "Development and Application of a Deep Learning Model for Predicting Sediment Transport in Agricultural Landscapes",
    bold: true, size: 30, font: FONT,
  })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 40 },
  children: [new TextRun({ text: "Halilu Naziru\u00B9* and Habibu Ismail\u00B9", size: 24, font: FONT })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 40 },
  children: [new TextRun({
    text: "\u00B9Department of Agricultural and Bio-Resources Engineering, Faculty of Engineering, Ahmadu Bello University, Zaria, Nigeria",
    italics: true, size: 20, font: FONT,
  })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 240 },
  children: [new TextRun({ text: "*Corresponding author", italics: true, size: 20, font: FONT })],
}));

// ================= Abstract =================
children.push(heading("Abstract", HeadingLevel.HEADING_1));
children.push(p("Gully erosion is a major threat to soil and water conservation in agricultural landscapes, and reliable prediction of sediment transport is essential for planning erosion-control interventions. Most existing predictive tools for small agricultural watercourses rely on simple empirical or single-variable linear relationships that struggle to capture the non-linear interactions among hydraulic and channel variables. This study develops and applies a deep learning model, a feedforward multilayer perceptron (MLP), to predict sediment transport rate in a gully system within the watercourse of Ahmadu Bello University Dam, Zaria, Nigeria, an agricultural landscape actively managed with a vegetative erosion-control measure (Morning Glory, Ipomoea carnea). Field data comprising 100 observations of flow depth, channel slope, soil shear stress and flow velocity, together with the corresponding sediment transport rate, were compiled from five field scenarios spanning pre-control and post-control conditions at two ponding depths (1.0 m and 1.5 m). These hydraulic observations were linked, spot by spot, to an independent channel survey recording channel breadth and average channel depth at the same 20 locations, giving a six-variable predictor set. The governing physical relationships underlying sediment transport in the study channel, including the Duboys bed-load equation and the shear-stress and continuity relations linking flow depth, slope and velocity, are set out alongside the mathematical formulation of the deep learning model. A two-hidden-layer MLP (8 and 4 neurons, ReLU activation) was trained on standardised inputs and evaluated using 10-times-repeated 5-fold cross-validation and an independent 80/20 hold-out split, then benchmarked against multivariate linear regression models and a four-variable version of the same deep learning model. The six-variable deep learning model achieved a mean cross-validated coefficient of determination (R\u00B2) of 0.62 (RMSE = 0.70 kg/s/m) and an R\u00B2 of 0.91 (RMSE = 0.43 kg/s/m) on the independent hold-out set, a clear improvement over the four-variable deep learning model (hold-out R\u00B2 = 0.62) and closely comparable to the multivariate linear benchmark (hold-out R\u00B2 = 0.93; cross-validated R\u00B2 = 0.78). Permutation importance analysis identified soil shear stress and channel average depth as the strongest predictors, followed by flow depth, channel breadth and slope, with flow velocity contributing least once the other variables were included. These results show that combining independently surveyed channel geometry with hydraulic measurements improves sediment transport prediction, and they illustrate both the promise and the current data-driven limitations of deep learning for small-scale agricultural erosion studies. The model, governing equations and workflow developed here provide a reproducible, physically grounded basis for scaling deep-learning-based sediment transport prediction to larger, multi-site agricultural datasets."));
children.push(p("Keywords: Deep learning; multilayer perceptron; sediment transport; gully erosion; Duboys equation; agricultural landscape; Ipomoea carnea; watercourse conservation", { italics: true }));

// ================= 1. Introduction =================
children.push(heading("1. Introduction", HeadingLevel.HEADING_1));

children.push(heading("1.1 Background", HeadingLevel.HEADING_2));
children.push(p("Gully erosion is one of the most visible and damaging forms of land degradation in agricultural landscapes, reshaping channels, removing productive topsoil and delivering large sediment loads to downstream water bodies (Poesen et al., 2003). Within agricultural watercourses, the rate at which sediment is transported is governed by an interacting set of hydraulic and geomorphic variables, including flow depth, channel slope, soil shear strength, flow velocity and channel geometry, and their combined effect is rarely a simple linear function of any one variable (Kirkby and Bracken, 2009). Reliable, site-specific prediction of sediment transport is nevertheless central to planning erosion-control measures, sizing conservation structures and evaluating the effectiveness of interventions such as vegetative cover."));
children.push(p("Traditional approaches to predicting sediment transport in small agricultural channels have relied heavily on empirical or physically based bed-load equations, most notably the Duboys (1879) shear-excess formulation, together with simple regression relationships between stream velocity and sediment load (Van Rijn, 1984a,b). While useful and easy to apply, such models generally assume a fixed, low-order functional form and are known to oversimplify the non-linear, multi-variable character of sediment transport processes, particularly where channel conditions change substantially, for example following the installation of an erosion-control measure. A companion field study in the same watercourse used a single-variable linear regression between stream velocity and sediment transport rate to evaluate the effectiveness of Morning Glory (Ipomoea carnea) as a biological control against gully erosion; while that model captured the broad trend, its single-variable, in-sample formulation could not exploit the full information contained in the field measurements of depth, slope, soil shear and channel geometry."));

children.push(heading("1.2 Deep Learning in Sediment Transport Prediction", HeadingLevel.HEADING_2));
children.push(p("Machine learning and, more recently, deep learning methods have been shown to substantially improve sediment transport prediction by learning non-linear relationships directly from data rather than assuming them a priori. Deep neural networks have outperformed conventional regression and other machine learning methods for predicting suspended and total sediment load in river systems (Shakya et al., 2023), and enhancement frameworks combining feature selection with deep neural networks have improved sediment discharge estimation accuracy in river basins by wide margins (Kaloop et al., 2025). Comparative studies have likewise found deep learning architectures such as long short-term memory (LSTM) networks to be more accurate than conventional artificial neural networks and support vector machines for sediment load prediction (Latif et al., 2023), and reviews of the field consistently note the growing role of deep learning relative to purely empirical or statistical models for erosion and sediment transport modelling (Andualem et al., 2023). Earlier work using machine learning approaches such as artificial neural networks and model trees also demonstrated clear gains over regression-based sediment rating curves once enough training data were available (Bhattacharya et al., 2007), while recent reviews emphasise that data volume, feature quality and appropriate validation protocols remain the key determinants of whether deep learning outperforms simpler alternatives in a given sediment transport application (Tao et al., 2021)."));
children.push(p("Despite this progress at river-basin scale, deep learning has rarely been applied to small, intensively monitored agricultural gully systems where field campaigns typically yield modest sample sizes but detailed, scenario-based measurements (e.g., before and after a control measure, at different ponding depths, and with an accompanying geometric channel survey). This study addresses that gap."));

children.push(heading("1.3 Objectives of the Study", HeadingLevel.HEADING_2));
children.push(p("Using field data collected from a gully system within the Ahmadu Bello University Dam watercourse, an agricultural landscape subject to active vegetative erosion control, this paper develops and applies a deep learning model, a feedforward multilayer perceptron, to predict sediment transport rate from flow depth, channel slope, soil shear stress, flow velocity, channel breadth and channel average depth. The specific objectives are to: (i) set out the governing physical equations of sediment transport relevant to the study channel and relate them to the field-measured variables; (ii) compile a multi-scenario field dataset spanning pre-control and post-control conditions, merged with an independent channel geometric survey; (iii) design, train and validate a deep learning model for sediment transport prediction, with a full mathematical formulation of the network and its training procedure; and (iv) benchmark the deep learning model against conventional multivariate linear regression models and a reduced four-variable version of the same network, in order to assess the added value, and the current limitations, of a deep learning approach at this data scale."));

// ================= 2. Materials and Methods =================
children.push(heading("2. Materials and Methods", HeadingLevel.HEADING_1));

children.push(heading("2.1 Study Area and Data Source", HeadingLevel.HEADING_2));
children.push(p("The study was carried out on a gully system within the watercourse of the Ahmadu Bello University (ABU) Dam, Zaria, Kaduna State, Nigeria, an agricultural landscape in the Nigerian Guinea Savanna. The site has been subject to active gully erosion and was managed using Morning Glory (Ipomoea carnea), a fast-establishing vegetative cover, together with controlled ponding of water outside the gully system as an erosion-control strategy. The field data used to develop the deep learning model in this paper were drawn from an earlier field campaign at this site that measured channel and flow variables before and after the vegetative control measure was established, at two ponding depths, together with an independent geometric survey of the channel."));

children.push(heading("2.2 Governing Equations of Sediment Transport", HeadingLevel.HEADING_2));
children.push(p("The field variables used in this study are linked by well-established governing relationships from open-channel hydraulics and sediment transport theory. These relationships motivate the choice of input variables for the deep learning model and provide a physical benchmark against which the data-driven model can be interpreted."));
children.push(p("The continuity equation relates discharge Q to flow velocity V, channel breadth B and flow depth D:"));
children.push(eq("Q = V \u00D7 B \u00D7 D", "1"));
children.push(eqNote("This equation motivates the inclusion of channel breadth as a predictor alongside flow depth and velocity, since, for a given discharge, breadth and depth jointly determine the flow velocity and hence the erosive energy available to transport sediment."));
children.push(p("The boundary (tractive) shear stress exerted by flowing water on the channel bed, following Straub (1935), is:"));
children.push(eq("\u03C4 = \u03B3 D S", "2"));
children.push(eqNote("where \u03C4 is the boundary shear stress, \u03B3 is the specific weight of water, D is flow depth and S is the channel slope. Equation (2) is the basis for the soil shear stress variable measured in the field and used as a model input."));
children.push(p("Duboys (1879) proposed that bed-load sediment is transported in stacked layers sliding over one another once the applied shear exceeds a critical threshold, giving the bed-load transport equation:"));
children.push(eq("q = k (\u03C4 \u2212 \u03C4c)", "3"));
children.push(eqNote("where q is the sediment transport rate, k is the Duboys coefficient (a function of grain size, obtained from the Duboys graph), \u03C4 is the boundary shear stress from Equation (2) and \u03C4c is the critical shear stress below which no sediment motion occurs, also read from the Duboys graph. Equation (3) was used in the field campaign to compute the sediment transport rate values that form the target variable of the deep learning model in this study."));
children.push(p("For reference, the Meyer-Peter and M\u00FCller (1948) bed-load formula, developed from extensive flume studies, expresses sediment transport as a power function of the excess shear stress:"));
children.push(eq("q\u2098 = C (\u03C4 \u2212 \u03C4c)^(3/2)", "4"));
children.push(eqNote("where q\u2098 is the bed-load transport rate and C is an empirical coefficient dependent on grain size and specific gravity; this formulation, together with the Shields (1936) and Schoklitsch (1934) criteria for incipient sediment motion, corroborates the shear-excess principle underlying Equation (3) and confirms that shear stress in excess of a critical value is the dominant physical driver of sediment transport in the type of channel studied here."));
children.push(p("Together, Equations (1)-(4) establish flow depth, channel slope, flow velocity, channel breadth and average channel depth as the physically relevant predictors of sediment transport rate in this study, and they justify the six-variable input set used to develop the deep learning model described in Section 2.6."));

children.push(heading("2.3 Field Data Collection", HeadingLevel.HEADING_2));
children.push(p("Flow depth was measured with a stadia rod, channel slope with a Garmin GPS unit and clinometer/altimeter, and flow velocity using the floating method. Soil shear stress was estimated from Equation (2), and the corresponding sediment transport rate was computed from Equation (3) using the Duboys shear-stress graph to obtain k and \u03C4c for the channel bed material. Measurements were collected under five scenarios: (i) pre-control (before any intervention); (ii) pre-control after ponding water outside the gully to a depth of 1.0 m; (iii) post-control after ponding at 1.0 m depth, i.e., after Morning Glory establishment; (iv) pre-control after ponding to 1.5 m depth; and (v) post-control after ponding at 1.5 m depth. Twenty paired observations of depth, slope, soil shear, sediment transport rate and flow velocity were recorded for each scenario, giving a total of 100 observations, listed in full in Appendix A."));

children.push(heading("2.4 Channel Geometric Survey", HeadingLevel.HEADING_2));
children.push(p("In addition to the scenario-based hydraulic measurements, an independent geometric survey of the channel was carried out over a 125 m reach at 20 fixed spot intervals, recording channel breadth and two depth readings (from which average channel depth was computed) at each spot, together with the local bed slope. Matching the recorded slope values confirmed that these 20 survey spots correspond exactly, spot for spot, to the 20 sample points (S/N 1-20) at which hydraulic variables were measured in each of the five field scenarios (maximum discrepancy in matched slope values across all 100 records was less than 2 \u00D7 10\u207B\u00B3, consistent with rounding), allowing channel breadth and average channel depth to be merged with the hydraulic dataset as two additional, scenario-invariant predictor variables consistent with the continuity relationship in Equation (1). Table 1 reproduces the channel survey data. The channel depth was found to peak at spot 18, attributed to poor grading of the channel bed at that location."));

children.push(caption("Table 1. Channel geometric characteristics along the 125 m surveyed reach (20 spot intervals)"));
children.push(dataTable(
  ["Spot", "Breadth (m)", "Depth 1 (m)", "Depth 2 (m)", "Average depth (m)", "Slope (%)"],
  [
    [1,"1.78","2.10","2.20","2.15","0.90"],[2,"1.95","2.20","2.01","2.10","0.67"],
    [3,"1.92","1.83","1.85","1.84","1.20"],[4,"2.01","1.95","1.90","1.92","0.95"],
    [5,"2.20","2.80","2.86","2.83","0.90"],[6,"2.30","2.50","2.54","2.52","0.70"],
    [7,"1.94","3.00","2.54","2.77","1.10"],[8,"2.70","2.20","2.23","2.22","0.76"],
    [9,"2.30","2.50","2.23","2.37","0.56"],[10,"2.20","2.30","2.31","2.31","2.10"],
    [11,"1.90","1.95","1.95","1.95","1.30"],[12,"3.00","1.67","1.65","1.66","1.40"],
    [13,"2.20","1.53","1.53","1.53","0.96"],[14,"1.98","1.65","1.96","1.96","0.96"],
    [15,"2.01","1.83","1.84","1.84","1.05"],[16,"2.23","1.76","1.67","1.72","1.12"],
    [17,"2.52","1.10","1.10","1.10","1.10"],[18,"3.35","1.23","1.20","1.22","1.51"],
    [19,"2.96","1.76","1.70","1.73","1.32"],[20,"2.81","2.01","2.01","2.01","1.08"],
  ],
  [1200, 1600, 1600, 1600, 2000, 1400]
));
children.push(new Paragraph({ spacing: { after: 200 }, children: [] }));

children.push(heading("2.5 Model Input and Output Variables", HeadingLevel.HEADING_2));
children.push(p("Six variables were used as model inputs: flow depth (ft), channel slope (dimensionless), soil shear stress (lb/ft\u00B2) and flow velocity (m/s) from the scenario-based hydraulic measurements, together with channel breadth (m) and channel average depth (m) from the geometric survey. The model output (target) variable was sediment transport rate, expressed in kg/s/m. Descriptive statistics for all variables are presented in Table 2. All input variables were standardised (zero mean, unit variance) prior to model training, using Equation (5) below, with scaling parameters computed only on each training fold to avoid information leakage into the corresponding test fold. To evaluate the contribution of the channel-geometry variables, a reduced four-variable model (hydraulic variables only) was also trained and evaluated under an identical protocol."));

children.push(caption("Table 2. Descriptive statistics of model variables (n = 100)"));
children.push(dataTable(
  ["Variable", "Mean", "Std. Dev.", "Minimum", "Maximum"],
  [
    ["Flow depth (ft)", "1.823", "0.847", "0.520", "3.600"],
    ["Channel slope (\u2212)", "0.0108", "0.0034", "0.0056", "0.0210"],
    ["Soil shear (lb/ft\u00B2)", "1.209", "0.585", "0.190", "2.620"],
    ["Flow velocity (m/s)", "1.147", "0.346", "0.220", "2.200"],
    ["Channel breadth (m)", "2.313", "0.428", "1.780", "3.350"],
    ["Channel average depth (m)", "1.988", "0.440", "1.100", "2.830"],
    ["Sediment transport rate (kg/s/m)", "1.476", "1.309", "0.030", "6.300"],
  ],
  [3200, 1600, 1600, 1600, 1600]
));
children.push(new Paragraph({ spacing: { after: 200 }, children: [] }));

children.push(heading("2.6 Mathematical Formulation of the Deep Learning Model", HeadingLevel.HEADING_2));
children.push(p("Each input variable x was standardised before being passed to the network:"));
children.push(eq("z = (x \u2212 \u03BC) / \u03C3", "5"));
children.push(eqNote("where \u03BC and \u03C3 are the mean and standard deviation of the variable computed on the training fold only."));
children.push(p("A feedforward deep neural network (multilayer perceptron, MLP) was developed with six input neurons (one per predictor variable), two hidden layers of 8 and 4 neurons respectively, using the rectified linear unit (ReLU) activation function, and a single linear output neuron for the predicted sediment transport rate. Figure 1 illustrates the network architecture. For a given hidden layer, the pre-activation value of neuron j is a weighted sum of the outputs of the previous layer:"));
children.push(eq("z\u2C7C = \u03A3\u1D62 w\u1D62\u2C7C a\u1D62 + b\u2C7C", "6"));
children.push(p("passed through the ReLU activation function:"));
children.push(eq("a\u2C7C = max(0, z\u2C7C)", "7"));
children.push(p("and the network output (predicted sediment transport rate) is a linear combination of the final hidden layer's activations:"));
children.push(eq("\u0177 = \u03A3\u2096 w\u2096 a\u2096 + b\u2092", "8"));
children.push(p("The network was trained by minimising a regularised mean squared error loss function:"));
children.push(eq("L = (1/n) \u03A3\u1D62 (y\u1D62 \u2212 \u0177\u1D62)\u00B2 + \u03B1 \u03A3 w\u00B2", "9"));
children.push(eqNote("where n is the number of training observations, y\u1D62 and \u0177\u1D62 are the observed and predicted sediment transport rates, w denotes all network weights and \u03B1 = 0.1 is the L2 regularisation coefficient used to limit overfitting given the modest sample size. The network was trained using the limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) optimisation algorithm, which is well suited to small datasets such as the one used here. A companion four-input version of the same architecture (hydraulic variables only, excluding channel breadth and average depth) was also trained for comparison, following the same Equations (5)-(9)."));
children.push(...figure("fig_network_architecture.png", "Figure 1. Feedforward multilayer perceptron architecture used for sediment transport rate prediction (6-4-8-1 configuration shown; the 4-variable model omits the two channel-geometry input nodes).", 480, 270));

children.push(heading("2.7 Model Training and Evaluation Protocol", HeadingLevel.HEADING_2));
children.push(p("Model performance was assessed using three complementary procedures. First, a 5-fold cross-validation was repeated 10 times with different random partitions (50 train/test folds in total) to obtain a stable estimate of average predictive performance and its variability, applied to both the six-variable and four-variable deep learning models and to matching multivariate linear regression models. Second, an independent 80/20 hold-out split (80 training observations, 20 test observations) was used to illustrate model performance on data not seen during training or hyperparameter selection. Third, permutation importance was computed on the hold-out set by measuring the drop in R\u00B2 when each input variable was independently shuffled 30 times, in order to rank the relative contribution of each predictor."));
children.push(p("Model performance was quantified using the coefficient of determination:"));
children.push(eq("R\u00B2 = 1 \u2212 [\u03A3\u1D62(y\u1D62 \u2212 \u0177\u1D62)\u00B2 / \u03A3\u1D62(y\u1D62 \u2212 \u1D67\u0305)\u00B2]", "10"));
children.push(p("the root mean square error:"));
children.push(eq("RMSE = \u221A[(1/n) \u03A3\u1D62 (y\u1D62 \u2212 \u0177\u1D62)\u00B2]", "11"));
children.push(p("and the mean absolute error:"));
children.push(eq("MAE = (1/n) \u03A3\u1D62 |y\u1D62 \u2212 \u0177\u1D62|", "12"));
children.push(p("Permutation importance for variable j was computed as the mean reduction in R\u00B2 across 30 random shuffles of that variable's column in the test set:"));
children.push(eq("I\u2C7C = R\u00B2\u2092\u1D63\u1D62\u1D4D\u1D62\u2099\u2090\u2097 \u2212 mean\u2081\u2080\u2083\u2080(R\u00B2\u209A\u2091\u1D63\u2098\u1D64\u1D62\u2091\u1D64,\u2C7C)", "13"));
children.push(eqNote("As a benchmark, a multivariate linear regression model using the same standardised input variables was evaluated under an identical protocol, and the single-variable velocity-only linear relationship used in the earlier field study of this site was also examined for reference. All analyses were implemented in Python 3 using the scikit-learn library."));

// ================= 3. Results =================
children.push(heading("3. Results", HeadingLevel.HEADING_1));

// ---- 3.1 EDA ----
children.push(heading("3.1 Exploratory Data Analysis", HeadingLevel.HEADING_2));
children.push(p("Before model development, the compiled dataset of 100 observations was examined to characterise the distribution of, and relationships among, the six predictor variables and the target sediment transport rate."));
children.push(...figure("fig_hist_target.png", "Figure 2. Distribution of sediment transport rate across all 100 field observations.", 400, 290));
children.push(p("Figure 2 shows that sediment transport rate is strongly right-skewed: most observations lie below 2 kg/s/m, corresponding to the pre-control, low-ponding-depth scenarios, while a smaller number of observations, mainly from the 1.5 m ponding scenarios, extend up to a maximum of 6.30 kg/s/m. This skew reflects the disproportionate erosive effect of higher ponding depths and motivated the use of cross-validation (rather than a single train/test split alone) to ensure that both the low- and high-transport regimes were adequately represented during model evaluation."));
children.push(...figure("fig_corr_heatmap.png", "Figure 3. Correlation matrix of the six model input variables and sediment transport rate.", 430, 380));
children.push(p("Figure 3 shows that soil shear stress and flow depth are the two variables most strongly correlated with sediment transport rate, consistent with their direct role in the Duboys and shear-stress equations (Equations 2-3), while channel slope, breadth and average depth show moderate positive correlation with the target and with each other, reflecting the fact that wider, deeper and steeper sections of the channel simultaneously experience higher shear stress and higher sediment transport. Flow velocity shows a comparatively weaker correlation with sediment transport rate than depth or shear, foreshadowing its low permutation importance reported in Section 3.4."));
children.push(...figure("fig_feature_scatter_grid.png", "Figure 4. Sediment transport rate plotted against each of the six predictor variables, with Pearson correlation coefficient (r) shown for each panel.", 500, 330));
children.push(p("Figure 4 confirms a broadly linear, positive relationship between sediment transport rate and each predictor, with soil shear stress (r = 0.90) and flow depth (r = 0.83) showing the tightest relationships and flow velocity (r = 0.55) the most scatter. None of the six relationships show pronounced curvature or reversal, which is consistent with the finding in Section 3.2 that a linear model performs competitively with the deep learning model on this dataset."));
children.push(...figure("fig_boxplot_scenario.png", "Figure 5. Distribution of sediment transport rate across the five field scenarios.", 420, 290));
children.push(p("Figure 5 shows that sediment transport rate increases substantially with ponding depth (comparing the 1.0 m and 1.5 m pre-control scenarios) and decreases after the Morning Glory control measure is established at each ponding depth (comparing the pre-control and post-control boxes at each depth), confirming that the five-scenario design captures a wide and physically meaningful range of channel conditions for model training."));

// ---- 3.2 CV performance ----
children.push(heading("3.2 Cross-Validated Model Performance", HeadingLevel.HEADING_2));
children.push(p("Table 3 summarises the repeated cross-validation performance of the six-variable and four-variable deep learning models alongside their multivariate linear regression counterparts."));
children.push(caption("Table 3. Repeated 5-fold cross-validation performance (mean \u00B1 standard deviation across 50 folds)"));
children.push(dataTable(
  ["Model", "R\u00B2", "RMSE (kg/s/m)", "MAE (kg/s/m)"],
  [
    ["Deep learning model, 6 variables", "0.62 \u00B1 0.37", "0.70 \u00B1 0.29", "0.31 \u00B1 0.10"],
    ["Deep learning model, 4 variables", "0.64 \u00B1 0.52", "0.59 \u00B1 0.49", "0.23 \u00B1 0.14"],
    ["Multivariate linear regression, 6 variables", "0.78 \u00B1 0.28", "0.50 \u00B1 0.23", "0.30 \u00B1 0.08"],
    ["Multivariate linear regression, 4 variables", "0.78 \u00B1 0.29", "0.50 \u00B1 0.24", "0.30 \u00B1 0.09"],
  ],
  [3800, 1400, 2200, 2200]
));
children.push(new Paragraph({ spacing: { after: 200 }, children: [] }));
children.push(...figure("fig_cv_r2_bar.png", "Figure 6. Cross-validated R\u00B2 (mean \u00B1 standard deviation) for each model.", 420, 300));
children.push(p("Figure 6 shows that the two multivariate linear regression models achieved the highest mean cross-validated R\u00B2 (0.78), followed by the four-variable deep learning model (0.64) and the six-variable deep learning model (0.62). Adding channel breadth and average channel depth to the deep learning model therefore did not improve, and slightly reduced, average cross-validated accuracy, while leaving the linear models essentially unchanged."));
children.push(...figure("fig_cv_r2_boxplot.png", "Figure 7. Distribution of fold-wise R\u00B2 values (50 folds per model) for all four models.", 440, 300));
children.push(p("Figure 7 shows that the deep learning models exhibit noticeably wider fold-to-fold variability in R\u00B2 than the linear models, including a number of folds with low or negative R\u00B2. This reflects the sensitivity of a flexible non-linear model to the particular 20-observation test fold drawn in a dataset of only 100 records, and it is the main reason the deep learning models' average cross-validated performance falls below that of the more stable linear benchmark."));
children.push(...figure("fig_cv_rmse_mae_bar.png", "Figure 8. Cross-validated RMSE and MAE (mean across 50 folds) for each model.", 440, 300));
children.push(p("Figure 8 shows the same ordering as Figure 6 in error terms: the linear regression models achieve the lowest mean RMSE and MAE, followed closely by the four-variable deep learning model, with the six-variable deep learning model showing the highest average error under cross-validation, consistent with its greater fold-to-fold variability."));
children.push(...figure("fig_cv_foldwise_line.png", "Figure 9. Mean R\u00B2 per cross-validation repeat (each point averages 5 folds) for the six-variable deep learning model and the six-variable linear regression model.", 440, 290));
children.push(p("Figure 9 tracks performance across the 10 independent repeats of 5-fold cross-validation. The linear regression model's repeat-wise mean R\u00B2 remains consistently high and stable (approximately 0.7-0.85 across all repeats), whereas the deep learning model's repeat-wise mean R\u00B2 fluctuates more widely (approximately 0.4-0.8), again illustrating that the flexibility of the neural network is only reliably realised with a larger and more diverse training set than the 80 observations available in each fold."));

// ---- 3.3 Hold-out ----
children.push(heading("3.3 Hold-Out Test Set Performance", HeadingLevel.HEADING_2));
children.push(p("Table 4 reports performance on the independent 80/20 hold-out split (80 training observations, 20 test observations), which provides a single, easily interpretable illustration of model performance on unseen data."));
children.push(caption("Table 4. Hold-out test set performance (80/20 split, n = 20 test observations)"));
children.push(dataTable(
  ["Model", "R\u00B2", "RMSE (kg/s/m)", "MAE (kg/s/m)"],
  [
    ["Deep learning model, 6 variables", "0.91", "0.43", "0.26"],
    ["Deep learning model, 4 variables", "0.62", "0.91", "0.38"],
    ["Multivariate linear regression, 6 variables", "0.93", "0.38", "0.29"],
  ],
  [3800, 1400, 2200, 2200]
));
children.push(new Paragraph({ spacing: { after: 200 }, children: [] }));
children.push(...figure("fig_holdout_pred_obs_6feat.png", "Figure 10. Predicted versus observed sediment transport rate for the six-variable deep learning model on the hold-out test set (n = 20). The dashed line indicates perfect agreement (1:1).", 400, 380));
children.push(p("Figure 10 shows that, on this particular 80/20 split, the six-variable deep learning model's predictions lie close to the 1:1 line across almost the entire observed range, including the higher sediment-transport observations (above 3 kg/s/m) that the four-variable model tended to under-predict, giving the model its strong hold-out R\u00B2 of 0.91."));
children.push(...figure("fig_holdout_residual.png", "Figure 11. Residuals (predicted minus observed) of the six-variable deep learning model plotted against the observed sediment transport rate.", 430, 320));
children.push(p("Figure 11 shows residuals scattered without a strong systematic trend across the observed range, though with slightly larger absolute residuals at the highest observed values, indicating the model neither substantially over- nor under-predicts as a function of transport magnitude, but that its absolute error grows somewhat for the largest, least frequent events."));
children.push(...figure("fig_holdout_error_hist.png", "Figure 12. Distribution of absolute prediction errors on the hold-out test set.", 410, 300));
children.push(p("Figure 12 shows that the majority of hold-out predictions (14 of 20) have an absolute error below 0.4 kg/s/m, with a small number of larger errors above 1 kg/s/m corresponding to the high-transport observations identified in Figure 11."));
children.push(...figure("fig_holdout_index_line.png", "Figure 13. Observed and predicted sediment transport rate for the 20 hold-out observations, ranked from highest to lowest observed value.", 440, 290));
children.push(p("Figure 13 confirms that the predicted series tracks the ranked observed series closely across the full range, with the largest deviations occurring at the single highest-transport observation, reinforcing that the model's largest errors are concentrated among rare, extreme events rather than distributed evenly across the dataset."));

// ---- 3.4 Importance ----
children.push(heading("3.4 Relative Importance of Predictor Variables", HeadingLevel.HEADING_2));
children.push(p("Table 5 reports permutation importance (Equation 13) for the six-variable deep learning model."));
children.push(caption("Table 5. Permutation importance of predictor variables for the six-variable deep learning model (hold-out set, 30 repeats)"));
children.push(dataTable(
  ["Variable", "Mean R\u00B2 drop", "Std. Dev."],
  [
    ["Soil shear (lb/ft\u00B2)", "0.824", "0.180"],
    ["Channel average depth (m)", "0.678", "0.370"],
    ["Flow depth (ft)", "0.372", "0.136"],
    ["Channel breadth (m)", "0.218", "0.193"],
    ["Channel slope (\u2212)", "0.180", "0.074"],
    ["Flow velocity (m/s)", "-0.027", "0.023"],
  ],
  [3800, 2200, 1600]
));
children.push(new Paragraph({ spacing: { after: 200 }, children: [] }));
children.push(...figure("fig_importance_bar.png", "Figure 14. Permutation importance (mean \u00B1 standard deviation) of each predictor variable for the six-variable deep learning model.", 440, 300));
children.push(p("Figure 14 shows soil shear stress as by far the most important predictor, consistent with its role as the direct driver of sediment transport in the Duboys equation (Equation 3), followed by channel average depth and flow depth. Channel breadth and slope contribute moderately, while flow velocity contributes negligibly and even slightly negatively once the other five variables are present, indicating its information content is largely redundant with depth and shear."));
children.push(...figure("fig_importance_4v6.png", "Figure 15. Permutation importance of the four hydraulic variables shared between the four-variable and six-variable deep learning models.", 440, 300));
children.push(p("Figure 15 compares the importance of the four shared hydraulic variables between the two model versions. Soil shear stress remains the dominant predictor in both models, but its estimated importance is substantially larger in the four-variable model (mean R\u00B2 drop of 3.16) than in the six-variable model (0.82), because in the absence of channel breadth and average depth the network relies more heavily on soil shear to explain variation that the six-variable model can partly attribute to channel geometry instead."));
children.push(...figure("fig_importance_pareto.png", "Figure 16. Pareto chart of predictor importance for the six-variable deep learning model, showing individual contribution (bars) and cumulative share (line).", 440, 300));
children.push(p("Figure 16 shows that the top two variables, soil shear stress and channel average depth, together account for the majority of the cumulative positive importance across all predictors, indicating that a simplified model built on these two variables alone could capture much of the predictive signal available in the full six-variable dataset, an option worth exploring in future work with a larger sample."));

// ---- 3.5 Comparison ----
children.push(heading("3.5 Comparison with Conventional Regression Approach", HeadingLevel.HEADING_2));
children.push(p("This section draws together the cross-validation and hold-out results to compare the deep learning and linear regression approaches directly."));
children.push(...figure("fig_grouped_bar_all_models.png", "Figure 17. Summary comparison of cross-validated R\u00B2, RMSE and MAE across all four models.", 560, 220));
children.push(p("Figure 17 restates the finding from Table 3 in a single comparative view: the linear regression models are marginally but consistently better than the deep learning models across all three cross-validated metrics, though the margin (R\u00B2 difference of 0.14-0.16) is modest relative to the fold-to-fold variability shown in Figure 7."));
children.push(...figure("fig_metrics_radar.png", "Figure 18. Normalised cross-validated performance profile of all four models across R\u00B2, inverse RMSE and inverse MAE (outer position indicates better performance).", 420, 420));
children.push(p("Figure 18 shows the two linear regression models occupying the outermost, most favourable position on all three normalised axes, with the four-variable deep learning model close behind and the six-variable deep learning model showing the most compressed (least favourable) profile under cross-validation, consistent with the pattern already identified in Table 3 and Figures 6-8."));
children.push(...figure("fig_comparison_scatter_overlay.png", "Figure 19. Hold-out predictions of the six-variable deep learning model and the six-variable linear regression model plotted together against observed sediment transport rate.", 420, 390));
children.push(p("Figure 19 shows that, on the hold-out split, both the deep learning model and the linear regression model track the 1:1 line closely and produce very similar predictions across the observed range, with the linear model marginally closer to the line at the highest observed value. This reinforces the overall conclusion that, once channel geometry is included, sediment transport rate in this gully system is predicted almost equally well by a flexible deep learning model and by a much simpler linear model, and that the value of the deep learning approach in this application lies chiefly in its ability to match, without requiring the analyst to pre-specify a functional form, the strong but largely linear relationship that governs this system, together with the flexibility to improve further as more field data become available."));

// ================= 4. Discussion =================
children.push(heading("4. Discussion", HeadingLevel.HEADING_1));
children.push(p("The deep learning model developed in this study successfully predicted sediment transport rate in an agricultural gully system from routinely measurable hydraulic variables and an independently surveyed channel geometry, achieving a cross-validated R\u00B2 of 0.62 and a hold-out R\u00B2 of 0.91 once channel breadth and average depth were added to the input set. This performance is broadly consistent with reports elsewhere that deep learning architectures can achieve strong predictive accuracy for sediment transport and discharge when applied to appropriately sized datasets (Kaloop et al., 2025; Latif et al., 2023), while also illustrating a well-documented limitation of deep learning: with only 100 field observations, average cross-validated performance of the deep learning model did not clearly exceed that of a much simpler multivariate linear regression model, even though the deep learning model performed best of all on the single hold-out split. This pattern, strong hold-out performance alongside more modest and variable cross-validated performance, is consistent with the broader finding in the sediment-modelling literature that the advantage of deep learning over conventional regression and machine learning approaches tends to grow with dataset size and variable complexity (Shakya et al., 2023; Andualem et al., 2023; Bhattacharya et al., 2007), and it suggests that the present dataset, while sufficient to demonstrate the workflow and the governing equations underlying it, is at the lower end of what is needed to fully and consistently exploit a deep network's non-linear capacity."));
children.push(p("The permutation importance analysis, and its correspondence with the governing equations set out in Section 2.2, offers additional insight of practical value. Soil shear stress and channel average depth emerged as the two strongest predictors of sediment transport rate, ahead of flow depth, channel breadth and slope, with flow velocity contributing least once the other variables were included, mirroring the shear-excess principle embedded in the Duboys equation (Equation 3), in which shear stress in excess of a critical threshold, not velocity per se, is the direct physical driver of bed-load transport. This suggests that, in this gully system, channel geometry, both the static average depth at a location and the dynamic soil shear generated there, carries more information about sediment transport than the instantaneous flow velocity alone, reinforcing the value of pairing hydraulic monitoring with a geometric channel survey rather than relying on velocity-based empirical relationships in isolation."));
children.push(p("From an agricultural landscape management perspective, the results are useful in two ways. First, the six input variables used, flow depth, channel slope, soil shear stress, flow velocity, channel breadth and channel average depth, are all readily measurable using simple field equipment (stadia rod, tape, GPS/clinometer and the floating method), so the modelling approach developed here, grounded in the governing equations of Section 2.2, could be applied by extension officers or watershed managers to estimate sediment transport rate at similar agricultural watercourses, either before deciding whether an intervention such as vegetative cover is warranted or to monitor the effectiveness of a measure already in place. Second, the clear improvement over the single-variable linear model previously used at this site (R\u00B2 rising from 0.19 to 0.62-0.93 depending on evaluation protocol) confirms that channel slope, soil shear and channel geometry carry information about sediment transport that velocity alone does not capture, reinforcing the case for combining hydraulic monitoring with a geometric channel survey in future field campaigns."));
children.push(p("Several limitations should be acknowledged. The dataset originates from a single gully system and covers a limited range of ponding depths and control conditions, so the model's ability to generalise to other agricultural watercourses, soil types or climatic settings has not been tested and should be treated with caution. The relatively small sample size also constrains the complexity of network architecture that can be reliably trained, and the repeated cross-validation results show considerable fold-to-fold variability, indicating that performance estimates, particularly the strong single-split hold-out result, should be interpreted as indicative of the model's potential rather than as a precise, generalisable accuracy figure. Expanding the field campaign to include multiple gully sites, additional seasons and a wider range of soil and vegetation conditions would provide the larger, more diverse dataset needed to determine more conclusively whether a deep learning model can decisively outperform simpler regression approaches in this application, consistent with recommendations in recent reviews of deep learning for sediment transport prediction (Andualem et al., 2023; Tao et al., 2021)."));

// ================= 5. Conclusion =================
children.push(heading("5. Conclusion and Recommendations", HeadingLevel.HEADING_1));
children.push(p("This study set out the governing physical equations of sediment transport relevant to a small agricultural gully system, the continuity relationship, the Straub shear-stress equation and the Duboys bed-load equation, and used them to guide the development of a deep learning model, a feedforward multilayer perceptron, for predicting sediment transport rate from field-measured flow depth, channel slope, soil shear stress, flow velocity, channel breadth and channel average depth. The model achieved a cross-validated R\u00B2 of 0.62 and a hold-out test R\u00B2 of 0.91, a marked improvement over a previously used single-variable linear model (R\u00B2 = 0.19), and closely comparable to, though not consistently better than, a multivariate linear regression benchmark built on the same variables. These findings indicate that sediment transport at this site is strongly related to the combination of measured hydraulic and geometric variables, in line with the governing equations presented, that a deep learning approach can capture this relationship well even with a modest field dataset, and that further gains from deep learning are likely contingent on expanding the size and diversity of the training dataset. It is recommended that future work extend field data collection to multiple agricultural gully systems and seasons, explore additional input variables such as soil particle size and vegetation cover density, and re-evaluate deep learning against conventional regression once a larger, multi-site dataset is available."));

children.push(heading("Acknowledgements", HeadingLevel.HEADING_1));
children.push(p("The authors thank the Department of Agricultural and Bio-Resources Engineering, Ahmadu Bello University, Zaria, for facilitating the field data collection on which this study is based."));

// ================= References =================
children.push(heading("References", HeadingLevel.HEADING_1));
const refs = [
  "Andualem, T. G., Hewa, G. A., Myers, B. R., Peters, S. and Boland, J. (2023). Erosion and sediment transport modeling: A systematic review. Land, 12(7), 1396. https://doi.org/10.3390/land12071396",
  "Bhattacharya, B., Price, R. K. and Solomatine, D. P. (2007). Machine learning approach to modeling sediment transport. Journal of Hydraulic Engineering, 133(4), 440-450.",
  "Duboys, P. (1879). Le Rh\u00F4ne et les rivi\u00E8res \u00E0 lit affouillable. Annales des Ponts et Chauss\u00E9es, 18(5), 141-195.",
  "Kaloop, M. R., Elsayed, M., Eldessouki, M., Hu, J. W., Lee, S.-J. and ELRashidy, N. (2025). Enhancement framework for modeling sediment discharge in rivers using novel supervised deep learning approaches. Journal of Soils and Sediments, 25(9), 2777-2796. https://doi.org/10.1007/s11368-025-04111-w",
  "Kirkby, M. J. and Bracken, L. J. (2009). Gully processes and gully dynamics. Earth Surface Processes and Landforms, 34(14), 1841-1851.",
  "Latif, S. D., Chong, K. L., Ahmed, A. N., Sherif, M., Sefelnasr, A. and El-Shafie, A. (2023). Sediment load prediction in Johor River: Deep learning versus machine learning models. Applied Water Science, 13, 79.",
  "Meyer-Peter, E. and M\u00FCller, R. (1948). Formulas for bed-load transport. Proceedings of the 2nd Meeting of the International Association for Hydraulic Structures Research, 39-64.",
  "Poesen, J., Nachtergaele, J., Verstraeten, G. and Valentin, C. (2003). Gully erosion and environmental change: Importance and research needs. Catena, 50(2-4), 91-133.",
  "Schoklitsch, A. (1934). Der Geschiebetrieb und die Geschiebefracht. Wasserkraft und Wasserwirtschaft, 29(4), 37-43.",
  "Shields, A. (1936). Anwendung der \u00E4hnlichkeitsmechanik und der Turbulenzforschung auf die Geschiebebewegung. Mitteilungen der Preussischen Versuchsanstalt f\u00FCr Wasserbau und Schiffbau, Berlin.",
  "Straub, L. G. (1935). Some observations on hydraulic and hydrologic engineering. Proceedings of the American Society of Civil Engineers.",
  "Tao, H., Al-Khafaji, Z. S., Qi, C., Zounemat-Kermani, M., Kisi, O., Tiyasha, T., Chau, K. W., Nourani, V., Melesse, A. M., Elhakeem, M. and Yaseen, Z. M. (2021). Artificial intelligence models for suspended river sediment prediction: State-of-the-art, modeling framework appraisal, and proposed future research directions. Engineering Applications of Computational Fluid Mechanics, 15(1), 1585-1612.",
  "Van Rijn, L. C. (1984a). Sediment transport, part I: Bed load transport. Journal of Hydraulic Engineering, 110(10), 1431-1456.",
  "Van Rijn, L. C. (1984b). Sediment transport, part II: Suspended load transport. Journal of Hydraulic Engineering, 110(11), 1613-1641.",
];
refs.forEach(r => children.push(refPara(r)));

// ================= Appendix A =================
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(heading("Appendix A: Full Field Dataset (100 Observations)", HeadingLevel.HEADING_1));
children.push(p("Table A1 lists the full set of 100 field observations used to train and evaluate the models in this study, comprising the five scenario blocks of 20 observations each (pre-control; 1.0 m pre-control; 1.0 m post-control; 1.5 m pre-control; 1.5 m post-control), merged with the corresponding channel breadth and average channel depth from the geometric survey (Table 1)."));

const scenarioLabelMap = {
  pre_control: "Pre-control", pond10_pre: "1.0 m pre-control", pond10_post: "1.0 m post-control",
  pond15_pre: "1.5 m pre-control", pond15_post: "1.5 m post-control",
};
const dataCsv = fs.readFileSync("data_full.csv", "utf-8").trim().split("\n").slice(1);
const appendixRows = dataCsv.map(line => {
  const parts = line.split(",");
  const [scenario, depth_ft, slope, soil_shear, sed_lb, sed_kg, vel, spot, breadth, chdepth] = parts;
  return [scenarioLabelMap[scenario], spot, depth_ft, slope, soil_shear, sed_kg, vel, breadth, chdepth];
});
children.push(caption("Table A1. Full field dataset (n = 100)"));
children.push(dataTable(
  ["Scenario", "Spot", "Depth (ft)", "Slope", "Shear (lb/ft\u00B2)", "Sed. rate (kg/s/m)", "Velocity (m/s)", "Breadth (m)", "Chan. depth (m)"],
  appendixRows,
  [1500, 700, 1100, 1000, 1300, 1300, 1100, 1100, 1200]
));

const doc = new Document({
  sections: [{
    properties: {
      page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } },
    },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("Deep_Learning_Sediment_Transport_Paper.docx", buf);
  console.log("done");
});
