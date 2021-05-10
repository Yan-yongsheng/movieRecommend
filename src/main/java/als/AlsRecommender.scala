package als
/*
为了更好地展示效果，这些代码我在Spark-shell中逐行运行的，而非直接将此函数运行
*/

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.sql.SparkSession
import org.jblas.DoubleMatrix
// 导入ALS推荐系统算法包
import org.apache.spark.mllib.recommendation.ALS

object alsRecommender {
  Logger.getLogger("org").setLevel(Level.WARN)
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster(Config.SPARK_CORES).setAppName("AlsRecommender")
    val sparkSession = SparkSession.builder().config(sparkConf).getOrCreate()
    import sparkSession.implicits._
    // 载入评分（1-5）数据
    val rawData = sc.textFile("/home/yys/movielens/ml-100k/u.data")
    // 展示第一条记录
    //rawData.first()

    // 格式化数据集,要加转义符
    val rawRatings = rawData.map(_.split("\\t").take(3))
    // 展示第一条记录
    rawRatings.first()

    // 导入rating类
    
    // 将评分矩阵RDD中每行记录转换为Rating类型
    val ratings = rawRatings.map { case Array(user, movie, rating) => Rating(user.toInt, movie.toInt, rating.toDouble) }
    
    ratings.first()

    
    // 启动ALS矩阵分解
    val model = ALS.train(ratings, 50, 10, 0.01)

    model.userFeatures.first
    //预测用户199对物品33的评分
    val predictedRating = model.predict(199,33)

    //为用户199推荐前15个物品
    val userId = 199
    //userId: Int = 199
    
    val K = 15
    //K: Int = 15

    val topKRecs = model.recommendProducts(userId, K)

    println(topKRecs.mkString("n"))


    // 导入电影数据集
    val movies = sc.textFile("/home/yys/movielens/ml-100k/u.item")
    // 建立电影id - 电影名字典
    val titles = movies.map(line => line.split("|").take(2)).map(array => (array(0).toInt, array(1))).collectAsMap()

    // 建立用户名-其他RDD，并仅获取用户199的记录
    val moviesForUser = ratings.keyBy(_.user).lookup(199)

    // 获取用户评分最高的10部电影，并打印电影名和评分值
    moviesForUser.sortBy(-_.rating).take(10).map(rating => (titles(rating.product), rating.rating)).foreach(println)


    //获取某用户推荐列表并打印：
    val topKRecs = model.recommendProducts(199, 10)
    topKRecs.map(rating => (titles(rating.product),rating.rating)).foreach(println)


    //下面这两行有些问题，没有运行出来
    // 基于电影隐特征，计算相似度矩阵，得到电影的相似度列表
    //val movieFeatures = alsModel.productFeatures.map{
    // case (movieId, features) => (movieId, new DoubleMatrix(features))
    }


    // 求向量余弦相似度
    //def consinSim(movie1: DoubleMatrix, movie2: DoubleMatrix):Double ={
    //movie1.dot(movie2) / ( movie1.norm2() * movie2.norm2() )
  }

}
