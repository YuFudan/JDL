# Comment

We appreciate the reviewer's time and remarks. We have provided more descriptions on dataset, designed extra baseline, reported comprehensive metrics and evaluation at your concern and suggestions.

## Strengths
  
* We also propose a spatial-temporal bias correction methods which involves non-trivial improvements on the existing methods in correcting waybill addresses and mining trajectory stay points.

## Weakness

1. Training and testing dataset need comprehensive descriptions.
   * What we have done:
     * We introduce the overall dataset at the beginning of 3\.2 and the testing ground truth in 4\.1\.1, while the statistical analysis is not comprehensive as you kindly remarked. Here we provide dataset descriptions in detail. We can add these descriptions in an appendix part of our paper later due to space constraints.
   * Overall statistics:
     * The dataset is collected in a logistics station of JD Logistics located in Tongzhou district, Beijing over a period of 37 days (from Aug. 1$^{st}$, 2022 to Sep. 6$^{th}$, 2022).
     * The data is composed of trajectory and waybills of each courier in each day, without any labels on the matching of waybills and stay points, or the labels on behaviors, except for the last day, when we send out data collectors to record the fine-grained behaviors. We use the data on the last day (Sep. 6$^{th}$) as the testing dataset and others as training dataset. The testing dataset is only one-day long since it is extremely costly to record the fine-grained behaviors.
     * We discard a courier's data in a day with lack of waybills or trajectory data. After this filtering, each courier has about 29 days of data and each day sees about 15 couriers. There are 15 couriers in the last day.
   * Statistical analysis:
     * As shown in the figure below, we report the statistics of our dataset from 4 aspects:
       * Number of waybills for each day
       * Number of different waybill addresses for each day
       * Number of stay points for each day
       * Number of candidate stay points per order (within 70m of order's Geocoding coordinate and 1h of order's confirmed finish time) for each day
      ![dataset statistics](https://cloud.tsinghua.edu.cn/f/2ee11cb32ee94a01bcde/?dl=1)
     * We observe that normally for each day, there are around 1500 waybills with 1000 addresses, 800 stay points, and 5 candidates stay points per order. There are fluctuations among different days, for example, in some day only a small portion of couriers are working. Nevertheless, The last day that serves as the testing dataset is not an outlier, which has around 2000 waybills, 1200 waybill addresses, 1200 stay points and 5 candidate stay points.
     * We also note that not all the couriers are used as ground truth in the last day, because only 4 couriers' fine-grained behaviors recorded by data collectors are of satisfactory granularity. So we also report the statistics for only these 4 couriers in the figure below:
      ![dataset statistics4](https://cloud.tsinghua.edu.cn/f/dc9ae5852728447aacc7/?dl=1)
     * From the figures, we draw that the data distribution of testing set generally consisit with training set, and it is not trivial with plenty of waybills and candidate stay points.
  
2. A stronger baseline is required to compare with the fine-grain model.
   * As we highlight in contributions, we are the first work to study the fine-grained courier behavior recovery, and we implement 2 baselines in the paper.
     * Existing works all assume the finish time of waybill to be the middle of a stay point, namely our baseline "mid".
     * We also design a baseline "unf" which assumes the finish time of waybil to be uniformly distributed during the stay interval.
   * Maybe these two baselines are not strong enough as you kindly remarked, so here we design an extra deep learning based baseline called "deep" and report its performance in the following table. (Note that we also extend the metrics in the performance table to each type of behavior as remarked in weakness 3, which will be discussed later.)
  ![metrics](https://cloud.tsinghua.edu.cn/f/92c58a2d697f4daca9b4/?dl=1)
   * Model design:
     * Note that we have no labels for the ground truth finish time of waybills in the training set. In order to train the NN, we use the confirmed finish time of waybills and adapt them into the duration of the matched stay point through linear transformation while keeping the ratio of gaps between the finish times.
     * The input features including the unit, floor of waybills, courier identity and a spatial grid index and we make the NN predict the time gap between any 2 consecutive waybills. The finish times based on NN prediction is again adapted into stay duration to avoid unreasonable results.
     * The corresponding fine-grained behaviors are decided based on the time gap of 2 waybills and their unit and floor number with empirical rules to allocate the time gap into behaviors including down-stairs, between-units, up-stairs.
   * Observations:
     * The "deep" fine-grained baseline generally achieves similar performance with "unf". Indeed, the "DTInf+deep" now replaces the "DTInf+unf" to be the best baseline w.r.t fine-grained accuracy, though their performances are still similar.
     * The reason why this baseline cannot address the problem well may be the fact that the confirmed finish time can be misleading, for example, couriers confirms a batch of waybills at once after finishing all of them, as mentioned in our paper.
  
1. The evaluation results are not comprehensive enough. Further evaluation of each behavior is needed.
   * What behaviors do we recover and what metrics do we report in the paper?
     * For coarse-grained behaviors, there are "deliver", "walk", "rest", and we report the overall accuracy and the accuracy of "deliver" since it is the most important behavior. The overall accuracy means the portion that the recovered behavior label is correct during the whole time axis throughout the courier's working process, while the "deliver" accuracy only check the time intervals when the courier is delivering (including fine-grained behaviors such as going upstairs).
     * For fine-grained behaviors, there are "walk", "rest", "up-stair", "down-stair", "between-unit", "deliver" and "arrange", where the "walk" and "rest" are the same with coarse-grained behaviors, and the coarse-grained "deliver" is further divided into "up-stair", "down-stair", "between-unit", "deliver" and "arrange". We only report the overall accuracy of fine-grained behaviors due to the various behaviors and limited space, though it requires no extra efforts for us to report all of them. Also, because the duration of fine-grained behaviors is quite short, if the duration of recovered result is basically correct only with some overall offsets, the accuracy can drop significantly. So maybe it is not necessary for high accuracy of each fine-grained behavior as long as the overall accuracy and the DTE metric is satisfactory. But still, we report the accuracy of each and every behavior in the table above.
   * How the baselines predict the behaviors?
     * For coare-grained behaviors, the interval of a stay point matched with waybills is "deliver", a stay point without waybills is "rest", and other time is "walk".
     * The "mid" baseline cannot actually figure out the fine-grained behaviors since it only predict the waybills' finish time to be the middle of stay interval.
     * The "unf" and "deep" baseline predicts different finish time for different waybills in a stay point, which implicitly suggest the fine-grained behaviors in between. We further check if the 2 consecutive waybills are in the same unit and on the same floor, and then the time gap between the 2 predicted finish times is uniformly allocated into "down-stair", "between-unit", "up-stair" and "deliver".
   * Observations:
     * For the coarse-grained behavior, our method achieves even larger improvement on "deliver" than overall accuracy, while the improvement on "rest" is smaller and the performance on "walk" is a little bit worse than "DTInf" (nearly the same).
     * For the fine-grained behavior, our method outperforms baselines significantly w.r.t each behavior except for "delivery" (a little bit worse than "MS+unf"). However, as mentioned before, the fine-grained behavior accuracy are vulnerable to offsets, and the accuracy of one single behavior can be misleading. Indeed, "MS+unf" actually performs quite bad on other behaviors though it is the best on "deliver". We highlight that it is more important to evaluate the overall fine-grained accuracy as well as the DTE.

## Detailed Suggestions

* We apologize for some minor typos. We will check and refine all the writing issues later on.

We appreciate your valuable remarks which can make this work more convincing on dataset and experiments. We humbly invite the reviewer to reconsider the merits of our work, which is the first to study the fine-grained courier behavior recovery problem, with our best efforts on baseline design and evaluating metrics from various aspects.