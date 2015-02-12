package com.medallia.word2vec;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.logging.Log;
import org.apache.thrift.TException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.medallia.word2vec.Searcher.Match;
import com.medallia.word2vec.Searcher.SemanticDifference;
import com.medallia.word2vec.Searcher.UnknownWordException;
import com.medallia.word2vec.Word2VecTrainerBuilder.TrainingProgressListener;
import com.medallia.word2vec.neuralnetwork.NeuralNetworkType;
import com.medallia.word2vec.thrift.Word2VecModelThrift;
import com.medallia.word2vec.util.AutoLog;
import com.medallia.word2vec.util.Common;
import com.medallia.word2vec.util.Format;
import com.medallia.word2vec.util.ProfilingTimer;
import com.medallia.word2vec.util.Strings;
import com.medallia.word2vec.util.ThriftUtils;

/** Example usages of {@link Word2VecModel} */
public class Word2VecExamples {
	private static final Logger logger = LoggerFactory.getLogger(Word2VecExamples.class);
	private static final Log LOG = AutoLog.getLog();
	
	private static final String inputFile = "stackexchange-travel-norm1-phrase1";
	private static final String modelFile = "stackexchange-travel-norm1-phrase1.model";
//	private static final String inputFile = "news.2012.en.shuffled-norm1-phrase1";
//	private static final String modelFile = "news.2012.en.shuffled-norm1-phrase1.model";
	
	/** Runs the example */
	public static void main(String[] args) throws IOException, TException, UnknownWordException, InterruptedException {
		demoWord();
//		loadModel();
	}
	
	/** 
	 * Trains a model and allows user to find similar words
	 * demo-word.sh example from the open source C implementation
	 */
	public static void demoWord() throws IOException, TException, InterruptedException, UnknownWordException {
		logger.trace("Entering demoWord()");
		
//		List<String> read = Common.readToList(new File("text8"));
//		List<String> read = Common.readToList(new File("stackexchange-travel-norm1-phrase1"));
//		List<String> read = Common.readToList(new File("news.2012.en.shuffled-norm1-phrase1"));
		List<String> read = Common.readToList(new File(inputFile));
		List<List<String>> partitioned = Lists.transform(read, new Function<String, List<String>>() {
			@Override
			public List<String> apply(String input) {
				return Arrays.asList(input.split(" "));
			}
		});
		
		Word2VecModel model = Word2VecModel.trainer()
				.setMinVocabFrequency(5)
				.useNumThreads(20)
//				.setWindowSize(8)
//				.setWindowSize(10)
				.setWindowSize(3)
				.type(NeuralNetworkType.CBOW)
				.setLayerSize(5)
//				.setLayerSize(100)
				.useNegativeSamples(25)
				.setDownSamplingRate(1e-4)
				.setNumIterations(1)
//				.setNumIterations(15)
				.setListener(new TrainingProgressListener() {
					@Override public void update(Stage stage, double progress) {
						System.out.println(String.format("%s is %.2f%% complete", Format.formatEnum(stage), progress * 100));
					}
				})
				.train(partitioned);
		
		try (ProfilingTimer timer = ProfilingTimer.create(LOG, "Writing output to file")) {
//			FileUtils.writeStringToFile(new File("text8.model"), ThriftUtils.serializeJson(model.toThrift()));
//			FileUtils.writeStringToFile(new File("stackexchange-travel-norm1-phrase1.model"), ThriftUtils.serializeJson(model.toThrift()));
//			FileUtils.writeStringToFile(new File("news.2012.en.shuffled-norm1-phrase1.model"), ThriftUtils.serializeJson(model.toThrift()));
			FileUtils.writeStringToFile(new File(modelFile), ThriftUtils.serializeJson(model.toThrift()));
		}
		
		interact(model.forSearch());
	}
	
	/** Loads a model and allows user to find similar words */
	public static void loadModel() throws IOException, TException, UnknownWordException {
		final Word2VecModel model;
		try (ProfilingTimer timer = ProfilingTimer.create(LOG, "Loading model")) {
//			String json = Common.readFileToString(new File("text8.model"));
//			String json = Common.readFileToString(new File("stackexchange-travel-norm1-phrase1.model"));
//			String json = Common.readFileToString(new File("news.2012.en.shuffled-norm1-phrase1.model"));
			String json = Common.readFileToString(new File(modelFile));
			model = Word2VecModel.fromThrift(ThriftUtils.deserializeJson(new Word2VecModelThrift(), json));
		}
		interact(model.forSearch());
	}
	
	/** Example using Skip-Gram model */
	public static void skipGram() throws IOException, TException, InterruptedException, UnknownWordException {
		List<String> read = Common.readToList(new File("sents.cleaned.word2vec.txt"));
		List<List<String>> partitioned = Lists.transform(read, new Function<String, List<String>>() {
			@Override
			public List<String> apply(String input) {
				return Arrays.asList(input.split(" "));
			}
		});
		
		Word2VecModel model = Word2VecModel.trainer()
				.setMinVocabFrequency(100)
				.useNumThreads(20)
				.setWindowSize(7)
				.type(NeuralNetworkType.SKIP_GRAM)
				.useHierarchicalSoftmax()
				.setLayerSize(300)
				.useNegativeSamples(0)
				.setDownSamplingRate(1e-3)
				.setNumIterations(5)
				.setListener(new TrainingProgressListener() {
					@Override public void update(Stage stage, double progress) {
						System.out.println(String.format("%s is %.2f%% complete", Format.formatEnum(stage), progress * 100));
					}
				})
				.train(partitioned);
		
		try (ProfilingTimer timer = ProfilingTimer.create(LOG, "Writing output to file")) {
			FileUtils.writeStringToFile(new File("300layer.20threads.5iter.model"), ThriftUtils.serializeJson(model.toThrift()));
		}
		
		interact(model.forSearch());
	}
	
	private static void interact(Searcher searcher) throws IOException, UnknownWordException {
		try (BufferedReader br = new BufferedReader(new InputStreamReader(System.in))) {
			while (true) {
				System.out.print("Enter word or sentence (EXIT to break): ");
				String word = br.readLine();
				if (word.equals("EXIT")) {
					break;
				}
				List<Match> matches = searcher.getMatches(word, 10);
				System.out.println(Strings.joinObjects("\n", matches));
				
				ImmutableList<Double> rawVector = searcher.getRawVector(word);
				System.out.println("Raw vector size: " + rawVector.size());
				System.out.println("Raw vector: " + StringUtils.join(rawVector, ", "));
				
//				for (Match match : matches) {
//					SemanticDifference similarity = searcher.similarity(word, match.match());
//					System.out.println("Similarity between " + word + " and " + match.match() + ": ");
//					for (Match m2 : similarity.getMatches(match.match(), 3)) {
//						System.out.println("===" + Strings.joinObjects("\n", matches));
//					}
//				}
			}
		}
	}
}
