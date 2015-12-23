package com.medallia.word2vec.util;

import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;

/** Helper to create {@link org.apache.log4j.NDC} for nested diagnostic contexts */
public class NDC implements AC {
	private static final Logger log = LogManager.getRootLogger();
	private final int size;

	/** Push all the contexts given and pop them when auto-closed */
	public static NDC push(String... context) {
		return new NDC(context);
	}

	/** Construct an {@link AutoCloseable} {@link NDC} with the given contexts */
	private NDC(String... context) {
		for (String c : context) {
			log.info("[" + c + "]");
		}
		this.size = context.length;
	}

	@Override
	public void close() {
	}
}
