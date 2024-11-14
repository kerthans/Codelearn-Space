public class TypeWriter {
    public static void slowPrint(String text, long millisPerChar) {
        for (char c : text.toCharArray()) {
            System.out.print(c);
            try {
                Thread.sleep(millisPerChar);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
        System.out.println();
    }
}