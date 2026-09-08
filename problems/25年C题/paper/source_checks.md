# 规则与引文核验记录

本任务是2026年对2025年C题的历史训练研究，不是2025年竞赛提交。不得称为已经通过官方提交审查。核验日2026-09-08。

## 格式来源

- 2025官方《报名和参赛须知》原件：https://www.mcm.edu.cn/upload_cn/node/745/ogFWOgXVc36c27bb3ee3adf21e06f53146b03585.pdf 。第4页要求电子论文无承诺/编号页、源代码在正文后附录，电子论文与支撑材料分开。确认其适用2025届次。
- 2025格式规范存档转载：https://www.cmathc.org.cn/mcm/tz/303.html ，2025-09-05标来源CUMCM。A4、至少2.5cm边距、摘要一页、电子稿首页摘要、无目录、正文尽量20页以内、附录不限；字体字号无统一要求。转载来源不是组委会域名，保留此来源限制。
- 官网当前页面 https://www.mcm.edu.cn/html_cn/node/4cd596519c9eb9fbd866398f6df0caa3.html 已是2026修订稿；本地根目录所给同名PDF也为2026，不作为2025的30页上限依据。官网链接的2020旧版页面返回404。

排版按已交叉确认的共同硬项以及2025存档的20页正文建议组织，不加入身份页；完整源码附录单列。Word可作为本任务完整参考论文交付，但不宣称其已满足任何当前真实竞赛的提交资格。模板使用skill内置Word构建基线，不伪称官方模板。

## 学术引用

双引擎检索已实际执行，完整结果位于reference_search/*.json；该检索只能发现候选，部分查询排在首位的是后来的综述，不能把这些条目误作目标原文。下列元数据另作核对：

1. Wang E, Batey A, Struble C, Musci T, Song K, Oliphant A. Gestational age and maternal weight effects on fetal cell-free DNA in maternal plasma. Prenatal Diagnosis, 2013, 33(7):662–666. DOI 10.1002/pd.4119。已打开Wiley原始页：https://obgyn.onlinelibrary.wiley.com/doi/10.1002/pd.4119 。仅用于方向性背景，不能移植其临床阈值性能到本附件。正文转述控制在来源许可范围内。
2. Laird NM, Ware JH. Random-effects models for longitudinal data. Biometrics, 1982, 38(4):963–974. DOI 10.2307/2529876。JSTOR原始出版入口已尝试但只返回访问页面；PubMed原始摘要及卷期页核验：https://pubmed.ncbi.nlm.nih.gov/7168798/ 。仅支持重复观测相关与随机效应框架，不声称本文复现其EM算法。
3. Yang J, Ding X, Zhu W. Improving the calling of non-invasive prenatal testing on 13-/18-/21-trisomy by support vector machine discrimination. PLOS ONE, 2018, 13(12):e0207840. DOI 10.1371/journal.pone.0207840。已打开原始出版页：https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0207840 。仅支持多指标判定的研究背景，本题未采用SVM，未引用其准确率为本题结果。
4. Efron B. Bootstrap Methods: Another Look at the Jackknife. The Annals of Statistics, 1979, 7(1):1–26. DOI 10.1214/aos/1176344552。已尝试原始Project Euclid链接但网页正文受限；CiNii/Crossref元数据及原文PDF搜索核对题名作者年份，未将检索返回的1992年导读当作1979原文。仅引基础重采样思想，主体簇重采样方案由本文自行定义，不声称此文证明本题小样本置信覆盖率。

受限来源不作逐字引用。本文主体推导和实验结论来自自身代码与数据，不能以未访问的原文支持新增复杂主张。
