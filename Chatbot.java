import java.util.*;

public class Chatbot {
    private static Map<String, List<String>> intents = new HashMap<>();
    private static Map<String, List<String>> responses = new HashMap<>();

    static {
        // Sample intents and responses
        intents.put("greeting", Arrays.asList("hello", "hi", "hey", "howdy", "greetings"));
        intents.put("farewell", Arrays.asList("bye", "goodbye", "see you later", "take care"));
        intents.put("weather", Arrays.asList("weather", "what's the weather like today", "how's the weather"));
        intents.put("unknown", Arrays.asList("I don't understand", "Can you please repeat that?", "Sorry, I can't help with that."));

        responses.put("greeting", Arrays.asList("Hello!", "Hi there!", "Hey, how can I assist you?"));
        responses.put("farewell", Arrays.asList("Goodbye!", "See you later!", "Take care!"));
        responses.put("weather", Arrays.asList("The weather is sunny today.", "It's raining outside.", "Expect a cloudy day today."));
    }

    public static List<String> preprocessText(String text) {
        List<String> stopWords = Arrays.asList("a", "an", "the", "is", "am", "are", "you", "I", "can", "please");
        List<String> words = Arrays.asList(text.toLowerCase().split("\\s+"));
        words.removeAll(stopWords);
        return words;
    }

    public static String getIntent(String userInput) {
        List<String> userWords = preprocessText(userInput);
        for (Map.Entry<String, List<String>> entry : intents.entrySet()) {
            String intent = entry.getKey();
            List<String> keywords = entry.getValue();
            for (String keyword : keywords) {
                if (userWords.contains(keyword)) {
                    return intent;
                }
            }
        }
        return "unknown";
    }

    public static String getResponse(String intent) {
        if (responses.containsKey(intent)) {
            List<String> responseList = responses.get(intent);
            return responseList.get(new Random().nextInt(responseList.size()));
        } else {
            List<String> responseList = responses.get("unknown");
            return responseList.get(new Random().nextInt(responseList.size()));
        }
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        System.out.println("Chatbot: Hi! How can I assist you?");
        while (true) {
            System.out.print("You: ");
            String userInput = scanner.nextLine();
            String intent = getIntent(userInput);
            String response = getResponse(intent);
            System.out.println("Chatbot: " + response);
        }
    }
}