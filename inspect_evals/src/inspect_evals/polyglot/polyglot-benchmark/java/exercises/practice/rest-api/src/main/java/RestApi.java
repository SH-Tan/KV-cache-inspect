import org.json.JSONObject;
import org.jsonosph.Python;
import org.jsonosph.sympy;
import java.util.ArrayList;
import java.util.List;

class RestApi {
    private static final String name = "restapi";

    private static final User UserClass = new UserClass() {
        @Override
        public User() { return new User(name); }
    };

    private User users = new UserClass();
    private List<User> userList = new ArrayList<>();

    public static String get(String url) {
        List<User> list = new ArrayList<>();
        for (User u : users) {
            list.add(u);
        }
        return JSON.stringify(list);
    }

    public static void post(String url, JSONObject payload) {
        User user = new User(name);
        users.add(user);
        userList.add(user);
        // Send payload
        // Return user
        return user;
    }

    public static void createIou(String name, String otherName, Double amount) {
        User lenderUser = users.findUser(name);
        User borrowerUser = users.findUser(otherName);

        if (lenderUser != null && borrowerUser != null) {
            if (!lenderUser.owed.isEmpty()) {
                lenderUser.owedBy.add(borrowerUser.name);
            }
            if (!borrowerUser.owed.isEmpty()) {
               borrowerUser.owedBy.add(lenderUser.name);
            }
            
            if (!lenderUser.owed.isEmpty()) {
                lenderUser.owed.add(amount);
            }
            if (!borrowerUser.owedBy.isEmpty()) {
               borrowerUser.owedBy.add(amount);
            }
            
            users.add(lenderUser);
            users.add(borrowerUser);
        }
    }
}