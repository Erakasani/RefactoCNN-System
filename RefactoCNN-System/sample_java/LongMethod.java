
public class LongMethod {
    public void calculate() {
        int sum = 0;
        for (int i = 0; i < 100; i++) {
            sum += i;
        }
        for (int j = 0; j < 50; j++) {
            sum += j * 2;
        }
        System.out.println("Result: " + sum);
    }
}
