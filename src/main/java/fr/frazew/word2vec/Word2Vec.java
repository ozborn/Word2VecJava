package fr.frazew.word2vec;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;

import org.apache.commons.io.FileUtils;
import org.apache.commons.logging.Log;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.thrift.TException;

import com.google.common.base.Function;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.medallia.word2vec.Searcher;
import com.medallia.word2vec.Searcher.Match;
import com.medallia.word2vec.Searcher.UnknownWordException;
import com.medallia.word2vec.Word2VecModel;
import com.medallia.word2vec.Word2VecTrainerBuilder.TrainingProgressListener;
import com.medallia.word2vec.Word2VecTrainerBuilder.TrainingProgressListener.Stage;
import com.medallia.word2vec.neuralnetwork.NeuralNetworkType;
import com.medallia.word2vec.thrift.Word2VecModelThrift;
//import com.medallia.word2vec.util.AutoLog;
import com.medallia.word2vec.util.Common;
import com.medallia.word2vec.util.Format;
import com.medallia.word2vec.util.ProfilingTimer;
import com.medallia.word2vec.util.Strings;
import com.medallia.word2vec.util.ThriftUtils;

public class Word2Vec {
	//private static final Log LOG = AutoLog.getLog();
	private static final Logger LOG = LogManager.getLogger();
	/** Runs the example */
	public static void main(String[] args) throws IOException, TException, UnknownWordException, InterruptedException {
		launch();
	}
	
	
	/** Loads a model and allows user to find similar words */
	public static void loadModelFromBin() throws IOException, TException, UnknownWordException {
		final Word2VecModel model;
		
		try (ProfilingTimer timer = ProfilingTimer.create("Loading model")) {
			File file = new File("knowledge-vectors-skipgram1000.bin");
			model = Word2VecModel.fromBinFile(file);
		}
		interact(model, model.forSearch());
	}
	
	/** 
	 * Trains a model and allows user to find similar words
	 * demo-word.sh example from the open source C implementation
	 */
	public static void launch() throws IOException, TException, InterruptedException, UnknownWordException {
		File f = new File("text8");
		File f2 = new File("text8.model");
		if (!f.exists() && ! f2.exists())
	       	       throw new IllegalStateException("Please download and unzip the text8 example from http://mattmahoney.net/dc/text8.zip");
		
		Word2VecModel model = null;
		if (f2.exists()) {
			model = Word2VecModel.fromThrift(ThriftUtils.deserializeJson(new Word2VecModelThrift(), FileUtils.readFileToString(f2)));
		} else {
			List<String> read = Common.readToList(f);
			List<List<String>> partitioned = Lists.transform(read, new Function<String, List<String>>() {
				@Override
				public List<String> apply(String input) {
					return Arrays.asList(input.split(" "));
				}
			});
			
			model = Word2VecModel.trainer()
					.setMinVocabFrequency(5)
					.useNumThreads(20)
					.setWindowSize(8)
					.type(NeuralNetworkType.CBOW)
					.setLayerSize(200)
					.useNegativeSamples(25)
					.setDownSamplingRate(1e-4)
					.setNumIterations(5)
					.setListener(new TrainingProgressListener() {
						@Override public void update(Stage stage, double progress) {
							System.out.println(String.format("%s is %.2f%% complete", Format.formatEnum(stage), progress * 100));
						}
					})
					.train(partitioned);
			
			try (ProfilingTimer timer = ProfilingTimer.create("Writing output to file")) {
				FileUtils.writeStringToFile(new File("text8.model"), ThriftUtils.serializeJson(model.toThrift()));
			}
		}
		
		interact(model, model.forSearch());
	}
	
	private static double getMean(double[] a) {
		double sum = 0.0D;
		for (int i = 0; i < a.length; i++) {
			sum += a[i];
		}
		return sum / a.length;
	}
	private static String arrayToString(double[] a) {
		String finale = "[";
		for (int i = 0; i < a.length; i++) {
			finale += a[i] + ",";
		}
		
		return finale + "]";
	}
	
	private static void normalize(double[] v) {
		double len = 0;
		for (double d : v)
			len += d * d;
		len = Math.sqrt(len);

		for (int i = 0; i < v.length; i++)
			v[i] /= len;
	}
	
	/** Loads a model and allows user to find similar words */
	public static void loadModel() throws IOException, TException, UnknownWordException {
		final Word2VecModel model;
		
		try (ProfilingTimer timer = ProfilingTimer.create("Loading model")) {
			String json = Common.readFileToString(new File("text8.model"));
			model = Word2VecModel.fromThrift(ThriftUtils.deserializeJson(new Word2VecModelThrift(), json));
		}
		interact(model, model.forSearch());
	}
	
	private static void interact(Word2VecModel model, Searcher searcher) throws IOException {
		try (BufferedReader br = new BufferedReader(new InputStreamReader(System.in))) {
			while (true) {
				System.out.print("Enter word or sentence (EXIT to break): ");
				String word = br.readLine().toLowerCase();

				if (word.equals("EXIT")) {
					break;
				}
				
				ExpressionParser parser = new ExpressionParser(word).parse();
				HashMap<String[], String> operations = parser.getTermsCombo();
				
				double[] finalVector = new double[model.layerSize];
				
				try {
					for (Entry<String[], String> entry : operations.entrySet()) {					
						double[] vector1 = null;
						double[] vector2 = null;
						
						if (entry.getKey().length == 1) {
							vector1 = finalVector;
							vector2 = searcher.getVector(entry.getKey()[0]);
						} else {
							vector1 = searcher.getVector(entry.getKey()[0]);
							vector2 = searcher.getVector(entry.getKey()[1]);
						}
						
						if (entry.getValue().equals(ExpressionParser.PLUS)) {
							if (finalVector == null) finalVector = searcher.getSum(vector1, vector2);
							else finalVector = searcher.getSum(finalVector, searcher.getSum(vector1, vector2));
						} else if (entry.getValue().equals(ExpressionParser.MINUS)) {
							if (finalVector == null) finalVector = searcher.getDifference(vector1, vector2);
							else finalVector = searcher.getSum(finalVector, searcher.getDifference(vector1, vector2));
						}
					}
					
					List<Match> matchesOld = searcher.getMatches(finalVector, 10);
					ArrayList<Match> matches = new ArrayList();
					for (Match match : matchesOld) {
						if (!parser.terms.contains(match.match())) matches.add(match);
					}
					
					System.out.println(Strings.joinObjects("\n", matches));
				} catch (UnknownWordException e) {}
			}
		}
	}
}

