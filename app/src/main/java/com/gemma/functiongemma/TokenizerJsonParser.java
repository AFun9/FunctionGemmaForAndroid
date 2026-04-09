package com.gemma.functiongemma;

import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonToken;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

final class TokenizerJsonParser {
    Map<String, Integer> parseVocab(File file) throws IOException {
        try (InputStream inputStream = new FileInputStream(file)) {
            return parseVocab(inputStream);
        }
    }

    Map<String, Integer> parseVocab(InputStream inputStream) throws IOException {
        try (Reader reader = new InputStreamReader(inputStream)) {
            return parseVocab(reader);
        }
    }

    Map<String, Integer> parseVocab(Reader input) throws IOException {
        Map<String, Integer> vocab = new HashMap<>();
        try (JsonReader reader = new JsonReader(input)) {
            reader.beginObject();
            while (reader.hasNext()) {
                String name = reader.nextName();
                if ("model".equals(name)) {
                    reader.beginObject();
                    while (reader.hasNext()) {
                        String modelName = reader.nextName();
                        if ("vocab".equals(modelName)) {
                            reader.beginObject();
                            while (reader.hasNext()) {
                                vocab.put(reader.nextName(), reader.nextInt());
                            }
                            reader.endObject();
                        } else {
                            reader.skipValue();
                        }
                    }
                    reader.endObject();
                } else {
                    reader.skipValue();
                }
            }
            reader.endObject();
        }
        return vocab;
    }

    List<String> parseMerges(File file) throws IOException {
        try (InputStream inputStream = new FileInputStream(file)) {
            return parseMerges(inputStream);
        }
    }

    List<String> parseMerges(InputStream inputStream) throws IOException {
        try (Reader reader = new InputStreamReader(inputStream)) {
            return parseMerges(reader);
        }
    }

    List<String> parseMerges(Reader input) throws IOException {
        List<String> merges = new ArrayList<>();
        try (JsonReader reader = new JsonReader(input)) {
            reader.beginObject();
            while (reader.hasNext()) {
                String name = reader.nextName();
                if ("model".equals(name)) {
                    reader.beginObject();
                    while (reader.hasNext()) {
                        String modelName = reader.nextName();
                        if ("merges".equals(modelName)) {
                            reader.beginArray();
                            while (reader.hasNext()) {
                                String mergeRule = null;
                                if (reader.peek() == JsonToken.STRING) {
                                    mergeRule = reader.nextString();
                                } else if (reader.peek() == JsonToken.BEGIN_ARRAY) {
                                    reader.beginArray();
                                    mergeRule = reader.nextString() + " " + reader.nextString();
                                    reader.endArray();
                                } else {
                                    reader.skipValue();
                                }
                                if (mergeRule != null) {
                                    merges.add(mergeRule);
                                }
                            }
                            reader.endArray();
                        } else {
                            reader.skipValue();
                        }
                    }
                    reader.endObject();
                } else {
                    reader.skipValue();
                }
            }
            reader.endObject();
        }
        return merges;
    }

    Map<String, Object> parseConfig(File file) throws IOException {
        try (InputStream inputStream = new FileInputStream(file)) {
            return parseConfig(inputStream);
        }
    }

    Map<String, Object> parseConfig(InputStream inputStream) throws IOException {
        try (Reader reader = new InputStreamReader(inputStream)) {
            return parseConfig(reader);
        }
    }

    Map<String, Object> parseConfig(Reader input) throws IOException {
        Map<String, Object> config = new HashMap<>();
        try (JsonReader reader = new JsonReader(input)) {
            reader.beginObject();
            while (reader.hasNext()) {
                config.put(reader.nextName(), parseValue(reader));
            }
            reader.endObject();
        }
        return config;
    }

    Map<String, Integer> parseAddedTokens(File file) throws IOException {
        try (InputStream inputStream = new FileInputStream(file)) {
            return parseAddedTokens(inputStream);
        }
    }

    Map<String, Integer> parseAddedTokens(InputStream inputStream) throws IOException {
        try (Reader reader = new InputStreamReader(inputStream)) {
            return parseAddedTokens(reader);
        }
    }

    Map<String, Integer> parseAddedTokens(Reader input) throws IOException {
        Map<String, Integer> addedTokens = new HashMap<>();
        try (JsonReader reader = new JsonReader(input)) {
            reader.beginObject();
            while (reader.hasNext()) {
                addedTokens.put(reader.nextName(), reader.nextInt());
            }
            reader.endObject();
        }
        return addedTokens;
    }

    private Object parseValue(JsonReader reader) throws IOException {
        switch (reader.peek()) {
            case BEGIN_OBJECT:
                Map<String, Object> obj = new HashMap<>();
                reader.beginObject();
                while (reader.hasNext()) {
                    obj.put(reader.nextName(), parseValue(reader));
                }
                reader.endObject();
                return obj;
            case BEGIN_ARRAY:
                List<Object> array = new ArrayList<>();
                reader.beginArray();
                while (reader.hasNext()) {
                    array.add(parseValue(reader));
                }
                reader.endArray();
                return array;
            case STRING:
                return reader.nextString();
            case NUMBER:
                String number = reader.nextString();
                try {
                    return number.contains(".") ? Double.parseDouble(number) : Long.parseLong(number);
                } catch (NumberFormatException e) {
                    return number;
                }
            case BOOLEAN:
                return reader.nextBoolean();
            case NULL:
                reader.nextNull();
                return null;
            default:
                reader.skipValue();
                return null;
        }
    }
}
